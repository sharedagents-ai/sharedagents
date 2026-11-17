# Release checklist

Steps for cutting a release of sharedagents.

## 1. Refresh the bundled models snapshot

The Models page ships a static fallback
(`src/sharedagents_web/data/models_snapshot.json`) so a fresh install with no
`ARTIFICIAL_ANALYSIS_API_KEY` still renders the full models table. Regenerate it
from live Artificial Analysis data before every release so the bundled data is
current:

```bash
python scripts/refresh_models_snapshot.py --apply
```

Run it **dry** first to preview the diff (model count + snapshot date), then
`--apply` to write. The script is idempotent — a re-run over unchanged data is a
byte-for-byte no-op. It requires `ARTIFICIAL_ANALYSIS_API_KEY` (env or
`~/.sharedagents/.env`).

## 2. Run the test suite

```bash
python -m pytest -q
```

Most tests are unit/integration tests that run anywhere. Files or cases that
need optional extras, a live server, Docker or a browser are marked (see
`[tool.pytest.ini_options]` in `pyproject.toml`) or guarded with
`pytest.importorskip`, so a clean checkout should be green.


## 3. Run the leak guards

The release must not ship deployment-specific or secret material:

```bash
python -m pytest -q tests/test_no_local_references.py
python scripts/check_forbidden_patterns.py dist/   # after §5's build
```

Run both with your private denylist set, so identity strings are checked as
well as the generic rules:

```bash
export SHAREDAGENTS_SCRUB_DENYLIST=~/.config/sharedagents/scrub-denylist.txt
python scripts/check_forbidden_patterns.py src docs scripts dist/
```

The denylist (names, handles, hosts; one per line, `#` comments,
case-insensitive substrings, accented names included) lives outside the repo
with mode 0600 and is never committed. `scripts/pypi_upload.sh` passes the
variable through. Without it the script warns
`no denylist given; only the generic rules apply` — treat that as a failed
release check.

## 4. Bump the version

The version is hand-maintained in **two** files. Bump both in the same commit —
they are not derived from one another:

| File | Line | What it holds |
| --- | --- | --- |
| `pyproject.toml` | `[project] version` | the packaged/distribution version |
| `src/sharedagents_core/__init__.py` | `__version__` | the version `sharedagents_core` reports at runtime |

Do **not** hand-edit `src/sharedagents/__init__.py`: it derives its version from
installed distribution metadata (`importlib.metadata.version("sharedagents")`,
with a `"0.0.0"` fallback for a source tree with no installed metadata).

Verify after editing:

```bash
grep -n '^version' pyproject.toml; grep -n '^__version__' src/sharedagents_core/__init__.py
```

## 5. Build and publish to PyPI

> **Publishing is a separate, explicitly-authorized act.** A published version
> number can never be reused — PyPI rejects the re-upload — so bump again
> rather than re-publishing.

```bash
mkdir -p build/release-source
git archive HEAD | tar -x -C build/release-source
python -m build --outdir dist build/release-source
scripts/pypi_upload.sh       # leak-scan dist/, then twine upload --skip-existing
```

Build from a clean Git archive of the release commit, never a developer's
working tree. The `sdist` settings in `setup.cfg` keep the archive's ownership
metadata neutral. Scan both distribution files against every secret value in
local environment/config files and the private identity denylist before
uploading; any hit blocks the release.

`scripts/pypi_upload.sh` requires `SHAREDAGENTS_PYPI_TOKEN` (env, or sourced from
`$SHAREDAGENTS_ENV_FILE`, default `~/.sharedagents/.env`), refuses to run with an
empty `dist/`, and runs `scripts/check_forbidden_patterns.py` over the built
artifacts before uploading.
