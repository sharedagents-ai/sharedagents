# Repository utilities

Run scripts from the repository root in the development environment.

| Script | Purpose |
| --- | --- |
| `check_forbidden_patterns.py` | Scan source directories or recursively inspect distributions for private patterns; accepts an external private denylist |
| `mkdocs_hooks.py` | Generate the documentation home from the README and rewrite repository links for the site |
| `capture_docs.py` | Capture shipped browser templates with isolated, labeled demo fixtures and no external requests |
| `check_website.py` | Check the built website's local links, mobile layouts, image switcher, dialog, and copy command |
| `refresh_models_snapshot.py` | Preview or explicitly refresh the bundled model catalog using the configured provider key |
| `pypi_upload.sh` | Scan and upload audited distributions using runtime release credentials |
| `reprice_turn_costs.py` | Inspect historical usage pricing; writes only with its explicit apply option |

Publication follows [RELEASES.md](../RELEASES.md). Never run a write operation
against a shared runtime as a diagnostic shortcut; inspect the target and use
a backup or disposable home. Review and redact any locally generated output
before attaching it to an issue or commit.
