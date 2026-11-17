## Why

<!-- What problem does this solve? Link the issue: Closes #N -->

## What changed

<!-- Bullet the notable changes -->

## Verification

<!-- Commands run + their results. A change you can't prove is unverified. -->

```bash
# e.g.
python -m pytest -q tests/test_<area>.py
```

## Checklist

- [ ] Tests added/updated for the behavior change
- [ ] `python -m pytest -q` is green (or failures explained)
- [ ] No deployment identity (hosts, IPs, home paths, personal emails) —
      `python -m pytest -q tests/test_no_local_references.py` passes
- [ ] Changelog entry under `## [Unreleased]` for user-facing changes
- [ ] Docs updated where behavior changed
- [ ] Logs, screenshots, fixtures, and generated assets contain no private data
- [ ] New dependencies and third-party assets include license/provenance details
