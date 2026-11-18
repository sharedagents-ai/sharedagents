# Project governance

SharedAgents uses a maintainer-led review process. Project ownership and review
responsibilities are recorded through the repository's maintainer permissions
and [CODEOWNERS](.github/CODEOWNERS), using project or team identities.

## Roles

Contributors submit issues, documentation, tests, and code through pull
requests. Reviewers assess correctness, maintainability, security, and scope.
Maintainers approve direction, merge changes, manage releases, and handle
private security and conduct reports.

Maintainership is based on sustained, constructive contributions and review
work. Access changes are a maintainer decision, with permissions limited to
the responsibilities being assigned. No contribution automatically grants
repository or production access.

## Decisions

Routine fixes can proceed through a focused pull request. Proposals involving
new top-level packages, large dependencies, public interfaces, storage formats,
licensing, or a broad redesign start with an issue describing the problem,
alternatives, migration, and verification plan.

Maintainers seek agreement through public technical discussion and record the
result in the issue or pull request. When agreement is not possible, a
maintainer responsible for the affected subsystem decides and explains the
tradeoff. Security-sensitive details remain in a private advisory until safe
to disclose.

## Reviews and releases

The default branch receives changes through pull requests and required checks.
Authors describe what changed and the evidence behind it; reviewers focus on
the submitted scope. Release work follows [RELEASES.md](RELEASES.md), including
privacy scans and explicit publication authorization.

Changes to governance follow the same proposal and review process. All
participation is subject to the [code of conduct](CODE_OF_CONDUCT.md).
