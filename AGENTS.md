# Rules for agents working in this repository

## Never commit local references or secrets

This repository and the PyPI package built from it are public-facing. Nothing
in them may identify a particular deployment or the person who runs it.

Never write any of the following anywhere in the repository: code, tests,
fixtures, docs, comments, built-in prompts, profile agents, skills, default
configs or package metadata.

- Secrets of any kind: API keys, tokens, OAuth client secrets, passwords,
  private keys.
- Host names, machine nicknames, IP addresses, internal domains or URLs.
- Home directories, usernames, SSH key names, email addresses, personal names.
- A specific deployment's agent or model lineup, or speeds and benchmarks
  measured on specific hardware.

Write generic code that reads such values at runtime. Deployment-specific
values live only in the user's sharedagents home (`~/.sharedagents/`: `config.json`,
`agents/<name>/config.json`, `agents/<name>/AGENTS.md`, `.env`), never in this
repository.

Tests and examples use placeholders: `example.com`, documentation IPs
(`192.0.2.x`, `203.0.113.x`), `user@example.com`, and fake tokens that cannot
be mistaken for real ones.

## Before publishing to PyPI

Build from a clean `git archive` of the release commit. Then scan the wheel
and the sdist for every secret value in the local env files and for identity
patterns: home paths, usernames, emails, private IPs, internal domains and
host names. Any hit blocks the release.
