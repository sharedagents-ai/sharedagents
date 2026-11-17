"""Skills bundled with the sharedagents package itself.

These are discovered by ``sharedagents_skill.skills.skill_search_paths`` at the
lowest precedence — a user-installed skill at ``~/.sharedagents/skills/<name>/``
overrides the bundled copy. The bundled copy is the source of truth that
ships with the codebase; it does not depend on anything in the user's
home directory existing before first use.

Skill state (venvs, sessions, on-disk profiles, downloaded artefacts)
still lives under ``~/.sharedagents/skills/<name>/`` because it's user-
mutable and per-machine. The skill's SKILL.md is responsible for
pointing at those state directories explicitly.
"""
