# Dev tab forge

The Dev tab and the SDLC tools open issues, push branches, open pull requests
and read CI status on a code forge. SharedAgents ships two backends, Gitea
and GitHub, behind one client (`sharedagents_web.forge_client`).

Set these in the environment or in the WebUI secrets dialog:

| Variable | Required | Meaning |
| --- | --- | --- |
| `FORGE_TOKEN` | yes | API token. It needs repo and issue scope. |
| `FORGE_KIND` | no | `gitea` or `github`. Inferred from `FORGE_URL` when unset: `github.com` means GitHub, any other host means Gitea. |
| `FORGE_URL` | Gitea: yes | Web root of the forge, for example `https://git.example.com`. GitHub defaults to `https://github.com`. When unset, it is read from the `origin` remote of the local source clone (`FORGE_SRC`, `SHAREDAGENTS_SRC` or the working directory). |
| `FORGE_API_URL` | no | API root override. Defaults to `<FORGE_URL>/api/v1` for Gitea and `https://api.github.com` for GitHub (`<FORGE_URL>/api/v3` for GitHub Enterprise). |
| `FORGE_USER_AGENT` | no | HTTP `User-Agent` for API calls. The default looks like a browser, because some forges behind Cloudflare reject other clients. |

Without a usable configuration the Dev tab shows why and creates no loops,
because every loop is bound to a forge issue.

## Older variable names

Setups made before the `FORGE_*` names keep working. These are read when the
matching `FORGE_*` variable is unset:

| Backend | Older names |
| --- | --- |
| Gitea | `GITEA_TOKEN`, `GITEA_BASE`, `GITEA_USER_AGENT`, `GITEA_SRC` |
| GitHub | `GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_BASE`, `GITHUB_API` |
| Kind | `SHAREDAGENTS_FORGE` |

When both a Gitea and a GitHub token are set under the older names and
`FORGE_KIND` is unset, Gitea wins.
