---
name: skill-builder
description: Best practices and conventions for building sharedagents skills including directory structure, SKILL.md format, Python code organization, .venv isolation, and testing
---

# Skill Builder — Best Practices

## Where to Put Skills

**User-created skills go in ONE place:** `~/.sharedagents/skills/<skill-name>/`

The sharedagents skill search chain (first match wins):
1. `SHAREDAGENTS_SKILLS_DIRS` env var (colon-separated override)
2. Per-agent: `<agent_home>/skills/<name>/` (agent-specific override)
3. **Shared (canonical): `~/.sharedagents/skills/<name>/`** ← put your skills here
4. Claude Code compat: `~/.claude/skills/<name>/` (legacy)
5. Plugin bundles: `~/.sharedagents/plugins/<name>/skills/`
6. Config `skills_dirs` in `~/.sharedagents/config.json` (additive, low precedence)
7. Built-in: `<package>/builtin_skills/` (lowest, shipped with sharedagents)

**Rule:** Put new skills in `~/.sharedagents/skills/<name>/` unless they are truly agent-specific.

## Skill Directory Structure

```
~/.sharedagents/skills/my-skill/
├── SKILL.md              # Required — the skill definition (YAML frontmatter + markdown body)
├── .venv/                # Recommended — isolated Python environment for this skill
├── scripts/              # Optional — helper scripts, CLI tools
│   ├── my_tool.py
│   └── requirements.txt  # Python deps for this skill
├── references/           # Optional — reference docs, examples
├── assets/               # Optional — static files, templates
└── tests/                # Optional — test files
```

## SKILL.md Format

```yaml
---
name: my-skill
description: One-line description used for search ranking and listing. Be specific about when to use this skill.
metadata:
  sharedagents:
    always: false          # true = inlined every turn (use sparingly), false = on-demand
---
```

After the frontmatter, write the skill body as markdown:

```markdown
# My Skill Title

## Overview
What this skill does and why it exists.

## When to Use
Clear trigger conditions — when should this skill be loaded?

## Core Pattern
The main workflow or pattern this skill implements.

## Quick Reference
Cheat sheet of commands, tools, or key facts.

## Implementation
Detailed step-by-step instructions.

## Common Mistakes
What to avoid, gotchas, and failure modes.
```

## .venv Isolation

**Every skill with Python code SHOULD have its own `.venv/`:**

```bash
cd ~/.sharedagents/skills/my-skill/
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
deactivate
```

Benefits:
- No dependency conflicts between skills
- Each skill controls its own Python environment
- Easier to update/rollback individual skills
- Agent can activate the right venv before running skill code

## Pre-built Code for Speed

When a skill involves repeated operations, put executable Python code in `scripts/`:

```
scripts/
├── my_tool.py            # Main tool — importable and executable
├── requirements.txt      # Dependencies
└── helpers.py            # Shared utilities
```

Make scripts executable with the correct shebang:
```python
#!/usr/bin/env python3
"""My skill's main tool."""

import sys
# ... code ...
```

The agent should run skill scripts via the skill's .venv:
```bash
~/.sharedagents/skills/my-skill/.venv/bin/python ~/.sharedagents/skills/my-skill/scripts/my_tool.py args
```

## CLI Scaffolding & Validation

Use `sharedagents skills create` to scaffold a fully structured skill:

```bash
# Basic skill
sharedagents skills create my-tool --description "Automate routine operations with my-tool"

# With isolated .venv
sharedagents skills create my-tool --venv
```

Validate a skill's frontmatter, script permissions, and dependencies with `sharedagents skills check`:

```bash
# Check a specific skill
sharedagents skills check my-tool

# Check all installed skills
sharedagents skills check
```

## Naming Conventions

- Skill directory names: `kebab-case`, lowercase, no spaces
- Skill names in frontmatter: match the directory name exactly
- Descriptions: action-oriented, mention the trigger (e.g., "Use when...")
- Keep descriptions under 150 words for discoverability

## Testing

Before deploying a skill:
1. Test `SkillSearch(query="select:<name>")` loads the skill correctly
2. Test `SkillSearch(query="<keyword>")` finds the skill via fuzzy search
3. Verify the skill body renders as expected markdown
4. If the skill has Python code, run a quick smoke test

## Token Budget

- Getting-started skills: <150 words in frontmatter description
- Frequently-loaded skills: <200 words total
- Always-active skills: minimize — every word costs tokens every turn
- Heavy reference material: keep on-demand, use `references/` folder
