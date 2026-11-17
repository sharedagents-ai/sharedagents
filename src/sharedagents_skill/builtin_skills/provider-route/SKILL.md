---
name: provider-route
description: Rank subscription engines and prepaid credits to pick which provider a subagent should run on. Load before choosing any subagent, local ones included.
metadata:
  sharedagents:
    always: false
    snapshot: provider-route
---

# Provider route

Load before selecting a subagent (build, plan, explore and general included).
The Live snapshot below is the source for quota and model pins.

## Choose in this order

1. An explicitly requested engine wins, if available, granted and within
   budget. MoA requests use the `moa` skill.
2. State the capability the task needs: reasoning difficulty, tools, context,
   images, permissions. Read the `model` column; `cfg:` is only the agent's
   saved selection, not proof of what ran. A family name is not an intelligence
   ranking, and any capable model may take any kind of task. If unsure whether
   a model can do it, say so.
3. Subscriptions below 100% pace beat local. Take the first capable, permitted
   agent in the "below target" list (lowest pace, then earliest reset), across
   families. Qwen 3.8 27B is a strong local fallback but does not outrank quota
   that would otherwise expire. Decide per subtask.
4. If none is below 100%, use local when capable; otherwise the best capable
   dispatchable external agent, and give its pace. Unknown pace is not zero:
   refresh before claiming spare allowance. Prepaid USD is not expiring quota
   and never gains priority.
5. Aim for 100% with real work only: never invent work, duplicate tasks, widen
   scope, exceed budgets or promise an exact reading. Exactly 100 is not below
   target.

## Reading the snapshot

- Pace is the share of a plan's longest limit (monthly, else weekly) used by
  its reset at the current rate. A 5-hour limit never changes pace; only a
  reached limit (`limit hit`) drops an agent.
- A plan with several pools (Cursor Auto and API) sums them by dollar limit:
  each pool's longest limit and its use are added in dollars, and pace is the
  summed use against the summed limit, so a small pool barely moves it.
  `pools:` gives each pool's own pace; a low one is allowance about to expire.
  `(equal weight)` marks a pool with no dollar amount, counted as an average
  pool. A pool marked `not counted` cannot be spent by this agent. The agent
  drops only when every counted pool is reached.
- Only dispatchable agents are spendable. "Recommended external" is the best by
  pace and can be above 100, which does not mean external beats local.
- The Usage-page winner can be blocked by a model pin or have no dispatchable
  agent. If your pick differs, name the blocker and the lowest-pace eligible
  alternative. Never rank by raw remaining percent.
- Locked or reserve-only quota does not authorize changing a model pin, and a
  per-call model override is not proof a different meter is used. Percent and
  USD scores are not comparable.
- A plan with no vendor CLI (MiMo) is a subscription, not a prepaid balance:
  spend it by delegating to the agent named in its row, like any other
  subscription engine.
- Answer "which engine" with agent, resolved model, task fit and pace first;
  remaining quota is supporting detail.
- Frontier delegations need `goal`, `constraints`, `acceptance_criteria` and a
  positive `budget_usd` within the authorized cap.

## Refresh

Cached about five minutes. Pass Skill args `refresh` when stale or after heavy
use. With a shell: `PYTHONPATH=src python3
src/sharedagents_skill/builtin_skills/provider-route/scripts/route.py [--refresh]
[--json]`. Jarvis has no Bash: use the snapshot below.
