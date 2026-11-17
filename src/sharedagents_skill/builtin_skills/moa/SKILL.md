---
name: moa
description: Mixture of agents (MoA / fusion). Use when the owner wants several engines to answer the same question, then critique each other and join into one answer.
metadata:
  sharedagents:
    always: false
---

# MoA

Same question to every engine, one critic-joiner, one answer. Do nothing else first: no `explore`, no `list`, no `provider-route`, no questions to the owner.

1. **Propose.** One `AgentAsync` call. Put the shared fields at the top level, once: `prompt` = the owner's question verbatim plus "Answer alone, concisely.", `goal`="Answer the question", `constraints`="Read-only", `acceptance_criteria`="A direct concise answer", `budget_usd`=1. Then `runs` = `[{"subagent_type":"claude"},{"subagent_type":"codex"}, …]`, one bare entry per external engine in the tool's Targets list (never `build`, `plan`, `explore`, `general`). Identical prompts are intended; the overlapping-topics rule does not apply. If the question concerns code, put the path in the prompt; the engines read it, you do not.
2. **Critique and join.** After every wave-1 result has arrived, one `AgentAsync` call to `claude`: the question, every answer labelled by engine, then "Find each answer's strongest flaw. Reply with only `FINAL:` one corrected answer, then `DISPUTE:` one line on where they disagreed." Same shared fields.
3. **Relay** that reply as written, plus one line naming the engines used and any dropped.

A refused or unavailable engine is dropped and named in that line. Never retry it or ask.
