# Deployment

## Foreground process

Install `sharedagents[web]` and configure the runtime as described in
[installation](../getting-started/installation.md). The stack command runs all
three services and reports their startup health:

```bash
python -m sharedagents.serve
python -m sharedagents.serve status
```

The first command stays in the foreground. Run the status command from another
terminal. Default addresses are:

| Component | Address |
| --- | --- |
| Orchestrator | `http://127.0.0.1:8099` |
| Gateway | `http://127.0.0.1:8100` |
| Browser UI | `http://127.0.0.1:8110` |

Ctrl+C stops the foreground stack. The launcher also supports `--detach` and
`stop`; these manage only a stack it launched. It is not a process supervisor.
See `python -m sharedagents.serve --help` for the supported options.

## Docker Compose

From a source checkout:

```bash
docker compose up --build -d
docker compose exec sharedagents sharedagents-web --create-user user@example.com
```

The image runs as a non-root user. Configuration, generated secrets, and other
state live in the `sharedagents-data` named volume at `/data`. Only the WebUI
is published, on host loopback by default. The gateway and orchestrator stay
on container loopback.

Provider variables can come from a private `.env` beside `compose.yml`, or
from the file named by `SHAREDAGENTS_ENV_FILE`. Start with `.env.example`, add
only the credentials you need, and keep the resulting file out of Git.
A model server on the Docker host is reachable through
`http://host.docker.internal:<port>/v1`.

## Direct WebUI startup

The `sharedagents-web` and `python -m sharedagents_web` entry points generate
or reuse a random `SHAREDAGENTS_WEB_SECRET` before loading the server. If you
embed the server through another WSGI entry point, supply a random secret from
your secret manager in the process environment before importing it. Do not put
its value on a command line or into source files. Create accounts with
`sharedagents-web --create-user`.

## Reverse proxies

Read [SECURITY.md](../../SECURITY.md) before publishing a port or configuring a
public tunnel. Use HTTPS and an appropriate access policy. Set
`SHAREDAGENTS_WEB_TRUSTED_PROXIES` only to the exact peer IP addresses the web
server sees. Do not trust arbitrary forwarded headers. Multi-hop proxy chains
need explicit review; the documented login throttle supports one trusted hop.

## Backups and upgrades

Back up the runtime home or named volume, not just the checkout. Store backups
privately because they include conversations and authentication state. Make
a consistent backup while the application is stopped or through a storage
snapshot that preserves SQLite consistency. Check release notes and validate
an upgrade against a separate runtime home before changing a shared deployment.
