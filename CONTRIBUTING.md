# Contributing to SharedAgents

Thanks for helping out. This file covers the local dev loop; for bug reports
and feature requests use the issue templates under
[Issues](https://github.com/sharedagents-ai/sharedagents/issues).

## Development setup

Python 3.10–3.14 is supported (the `[whatsapp]` extra needs 3.11+).

```bash
git clone https://github.com/sharedagents-ai/sharedagents.git
cd sharedagents
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,web,vllm-router]'   # add docs for the documentation site
pip install pre-commit build              # git hooks + packaging (not in [dev])
pre-commit install                        # optional: run the gates on every commit
```

`./sharedagents` does the venv + editable install for you and starts the TUI.
Runtime deps live in `src/requirements.txt` (kept in sync with
`pyproject.toml`); test/lint tooling lives in the `dev` extra. Optional
extras: `full`, `web`, `stt`, `bedrock`, `whatsapp`, `vllm-router`, `docs`.

## Checks

These are the commands CI runs; a PR should pass all of them locally.

```bash
# Tests (the default suite; e2e/docker/browser tests need a live stack)
python -m pytest -q -n auto -m "not e2e and not docker and not browser"

# Lint and formatting (ruff is pinned in the dev extra and pre-commit)
ruff check src tests scripts
ruff format --check src tests scripts     # `ruff format src tests scripts` to fix

# Leak guards (set SHAREDAGENTS_SCRUB_DENYLIST to your private list, see below)
python -m pytest -q tests/test_no_local_references.py
python scripts/check_forbidden_patterns.py src docs scripts

# Documentation
pip install -e '.[docs]' && mkdocs build --strict

# Packaging
python -m build && python scripts/check_forbidden_patterns.py dist
```

Useful variations: `python -m pytest -q tests/test_<area>.py` for one file,
`ruff check --fix` for the safe autofixes.

- Keep the suite lean and fast: the default run should finish in about a
  minute. Add a test for observable behavior (a reproduced bug, an API or
  security contract, the turn loop), not for mock wiring, prompt wording or
  private helpers, and retire a weaker test when a new one covers it.
- Tests that need optional extras guard with `pytest.importorskip`; tests
  needing a live stack, Docker or a browser are marked `e2e`, `docker` or
  `browser` and do not run in the default suite.
- `tests/test_e2e_live_model_reply.py` checks real model replies against a
  running WebUI. It is opt-in, because it spends provider tokens; set
  `SHAREDAGENTS_E2E_URL`, `SHAREDAGENTS_E2E_EMAIL`,
  `SHAREDAGENTS_E2E_PASSWORD_FILE` and `SHAREDAGENTS_E2E_MODELS` (aliases,
  comma-separated) and run `python -m pytest -m e2e`.
- **Never touch the operator's home in tests.** Redirect with
  `SHAREDAGENTS_HOME`/`tmp_path`; see `tests/conftest.py` for the pattern.
- Leak guards: `tests/test_no_local_references.py` scans the whole tree for
  host names, private IPs, home paths and personal emails, and
  `scripts/check_forbidden_patterns.py` does a byte-level pass over sources
  and build artifacts. Maintainers keep a private denylist of identity
  strings (names, handles, hosts) outside the repository, one per line,
  `#` for comments, matched case-insensitively as substrings, e.g. in
  `~/.config/sharedagents/scrub-denylist.txt` (mode 0600), and point
  `SHAREDAGENTS_SCRUB_DENYLIST` at it (or pass `--denylist FILE`); the check
  and `scripts/pypi_upload.sh` use it whenever it is set. Never commit the
  list or its terms. New identity-like fixtures must use placeholders —
  doc IPs (`192.0.2.x`, `203.0.113.x`), `example.com`, `user@example.com`,
  `*@sharedagents.ai`, obviously fake numbers.

## Code style

- Formatting is `ruff format` (Black-compatible, 120 columns); imports are
  sorted by ruff's isort rules. Don't hand-format.
- The enabled lint rules and every ignored rule, each with its reason, live in
  `[tool.ruff.lint]` in `pyproject.toml`. Prefer fixing a finding over adding
  a `noqa`; when a `noqa` is needed, say why on the same line.
- Every module starts with a docstring that says what it is for.
- Comments explain *why* the code is the way it is. Keep issue numbers,
  reviewer names, dates and incident narratives in commit messages and PR
  descriptions, not in the source.
- `.editorconfig` sets the whitespace rules for non-Python files.

## Pull requests

- One focused change per PR; describe the *why*, not just the *what*.
- Add or update tests for behavior changes — a change you can't prove is
  unverified.
- Keep `src/requirements.txt` and `pyproject.toml` dependencies in sync.
- Note user-facing changes under `## [Unreleased]` in `CHANGELOG.md`.

## Versioning

Releases follow SemVer, cut from `main` by maintainers per `RELEASES.md`.
The version lives in `pyproject.toml` and
`src/sharedagents_core/__init__.py` — bump both together.

## A note on scope

SharedAgents deliberately ships a lean core. New heavy dependencies and new
top-level packages need a reason — discuss in an issue first.
