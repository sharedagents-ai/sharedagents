# Maintainer guide

## Review and merge

Use feature branches and pull requests. Require CI and review before merging
into the default branch; keep a focused change small enough to review.
Security changes need explicit consideration of unauthenticated access,
cross-user access, failure paths, and whether logs expose sensitive values.

Document the observable change, relevant tests, migrations, and known limits in
the pull request. Do not treat a skipped live test as a successful check.
Resolve disagreements through the [governance process](../../GOVERNANCE.md).

## Privacy of public contributions

Review file contents and metadata, including images, generated assets, example
configs, and test captures. Inspect image pixels as well as EXIF/metadata:
text scanners cannot detect conversation content or identities in a screenshot.
Use only a sanitized demo runtime for screenshots. Preserve upstream license notices, but replace
deployment-specific identities with reserved example domains, documentation
IP ranges, and generic user names. Reproduction details belong in a small,
sanitized fixture rather than a production session dump.

Generic CI checks supplement the publisher's private denylist and exact-value
credential scan. Keep the denylist and scan output outside published files.
Do not disable a guard just because generated local caches fail it; use a clean
archive for publication.

## Public source imports

A first public import from a private development repository uses a reviewed
snapshot and fresh public history. It does not mirror private branches, tags,
commit authors, messages, remotes, or Git configuration. Retain legally required
attribution in the source and third-party notices.

Create public commits with project-owned identity metadata and check every ref
that will be pushed. A public-history rewrite after collaboration has started
requires a separate coordinated decision; ordinary contributions use normal
pull requests and preserve public contributor history.

## Releases

Follow [RELEASES.md](../../RELEASES.md). Update both version declarations,
changelog, and bundled data when required. Build from a clean archive, scan the
wheel and sdist, and verify an installation of the built package.

PyPI publication is a distinct authorized operation. After upload, verify the
index/download hashes against the audited artifacts and install from PyPI into
a fresh environment. Keep release credentials in the approved runtime secret
store; do not put them in workflow YAML, command arguments, or repository files.

## CI and repository settings

The GitHub workflow tests supported Python versions on Linux, macOS, and
Windows, plus separate lint, docs, and build jobs. Workflow tokens use read-only
repository access. Protect the default branch, enable security reporting and
available secret protection, and keep issue labels and review ownership aligned
with the actual maintainer team. Prefer configuration with no live credentials
or deployment assumptions.
