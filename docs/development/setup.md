# Development setup

## Checkout and environment

Fork the project for contributions from outside the maintainer team, then work
on a feature branch. Python 3.10–3.14 is supported; Python 3.12 is a convenient
local starting point.

```bash
git clone https://github.com/sharedagents-ai/sharedagents.git
cd sharedagents
git switch -c feature/short-description
python -m venv .venv
```

Activate the environment with `. .venv/bin/activate` on POSIX or
`.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,web,vllm-router,docs]"
python -m pip install build twine pre-commit
pre-commit install
```

The developer extra includes optional channel test dependencies. WhatsApp
requires Python 3.11+ and system libmagic; a lean contribution can run the
relevant unit tests without exercising that channel.

## Isolate runtime data

Use a disposable home when manually exercising the application. For example,
set `SHAREDAGENTS_HOME` to a private directory outside the checkout. Ensure
`SHAREDAGENTS_AGENTS_HOME` does not point at another installation. Never use
your real conversation store or credentials as test fixtures.

Automated tests use `tmp_path` and `tests/conftest.py` to isolate runtime files.
The [testing guide](testing.md) describes the selections and required checks.

## Find the owner of a change

Use the [architecture map](architecture.md) before adding a new package or
changing a cross-service interface. Browser templates and static files are
inside `src/sharedagents_web/`; they are runtime package data and must continue
to work from an installed wheel.

## Preview documentation

```bash
mkdocs serve
mkdocs build --strict
```

`README.md` is also the generated documentation home page. Keep links relative
to their source files so they work on GitHub; the MkDocs hook rewrites the
README's repository links for the site.

## Capture interface images

The public screenshots show real packaged templates and JavaScript with explicit
demo fixtures. They do not connect to a running installation, provider account,
or conversation store. To regenerate them:

```bash
python -m pip install -e ".[web]" playwright
python -m playwright install chromium
python scripts/capture_docs.py
```

The script blocks external browser requests, checks page errors, and labels every
capture **DEMO DATA**. Inspect every image visually and check its metadata and
text before committing it. Keep account balances, histories, user paths, session
titles, and deployment-specific agent or model selections out of public images.
Use fictional fixtures instead of screenshots from a working installation.

The [website build](../../website/README.md) combines the landing page and the
documentation in one static output.

## Prepare a pull request

Keep the diff focused and add regression coverage for behavior changes. Run
the relevant checks, update documentation, and describe the observed problem
and resulting behavior. For user-facing changes, add an entry under
`Unreleased` in `CHANGELOG.md`. Follow the review expectations in
[CONTRIBUTING.md](../../CONTRIBUTING.md).
