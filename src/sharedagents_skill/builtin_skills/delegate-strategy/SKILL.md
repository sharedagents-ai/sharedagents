---
name: delegate-strategy
description: Choose between local execution and a subscription delegate using task intelligence requirements and live subscription pace. Use before choosing local because it is cheap, or when deciding whether to escalate difficult work.
---

# Delegate by capability and pace

Do not assume the current agent runs on a small local model. Check the resolved
model and available tools. Load `provider-route` before selecting a child.

1. Honor an explicitly requested engine within grants and budgets.
2. Determine the intelligence and capabilities the actual task needs.
3. Among capable, permitted agents, use the first agent in the
   snapshot's "below target" list: subscriptions below 100% take priority over local inference.
   Local Qwen's low marginal cost is not a reason to leave usable allowance
   expiring. Select independently for each subtask, across model families.
4. If none qualify, execute locally when capable. Otherwise use a capable
   available external model and explain the pace tradeoff.

Use the resolved model in the snapshot's `model` column, not vendor stereotypes or code
line counts, to assess task fit. Missing pace is unknown, not zero. Refresh
stale quota before claiming headroom. Target 100% with requested work only;
never pad, duplicate or invent tasks to use allowance, and never bypass budgets.

For difficult or repeatedly failing work, reassess capability before retrying.
Pass the delegate the goal, constraints, acceptance criteria, authorized cost
budget, relevant evidence, attempted fixes and exact verification commands.
Use the available AgentSync/AgentAsync tool according to the caller's execution
mode; a foreground caller waits for completion. Existing grants, secret checks,
model-pin boundaries and preflight requirements continue to apply.

For parallel independent subtasks, choose each capable delegate from the live
pace ranking. Do not duplicate the same task merely to spend more quota.
