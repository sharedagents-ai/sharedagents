# Async delegation waves

Jarvis's native local-LLM harness requires a saved, non-empty todo list before
task tools run. Todo updates, owner questions, and child list/abort/steer
controls remain available without a plan. A plan from an earlier turn can be
reused. A failed todo write cannot unlock task execution.

Each physical turn may dispatch one delegation wave. Put all independent work
for the current phase in one
`AgentAsync(runs=[...])` call. The harness serializes Jarvis's top-level tool
calls so a todo write settles before the next action, and a dispatch settles
before another dispatch is considered; children inside the single batch still
run concurrently. A second dispatch in the same turn, including an attempted
append, is refused before the tool runner executes. Failed dispatches that
start no children can be corrected; partially successful batches consume the
wave. These Jarvis coordinator rules do not change worker tool execution,
explicit operator direct dispatch, or the underlying wave-storage API.

A wave may mix fresh children and continued ones. A run that carries a
finished child's `task_id` and a new prompt continues that child session, with
everything it already learned, as a member of the new wave. Jarvis is told to
prefer that over a fresh child whenever the work is the same task.

Every wave records the owner prompt it works for (its *origin*, stored in the
wave manifest). An owner prompt starts a new origin. A turn that only delivers
wave results inherits the origin of the delivered wave, so it can dispatch
that request's next wave without another owner prompt. Dependent plan, build,
and fresh review phases use successive waves; they must not run together in
one batch. The open-wave check compares origins:

- A new owner prompt may dispatch its wave even while waves from earlier
  prompts are still running. Jarvis is shown those live children first and
  told to steer them when the new message is about the same subject.
- A turn whose own request still has an open wave cannot dispatch another;
  it waits, or steers a live child.
- A turn with no known origin (heartbeats, goal or loop drives, waves written
  before origins existed) keeps the strict rule: no new wave while any wave of
  the session is open.

One agent per browser profile: a dispatch whose brief drives the browser
(browser, Chrome, nodriver, the live profile, an X display) is refused while
another live child of the session does browser work. Jarvis steers that child
or continues it after it finishes. Briefs that only say to stay out of the
browser do not count. The browser skill also holds a per-profile lease (see
the nodriver-browser skill).

After dispatch Jarvis receives a reply round to explain the actual dispatch and
its expected consolidated output, even if it already narrated before calling
the tool. This is the normal post-dispatch response, not a pending-todo nudge.
Results that arrive during that reply stay in the durable follow-up queue for
a fresh turn; they cannot consume the next phase inside the dispatching turn.
When Jarvis instead stops without any delegation and still has pending todos,
the harness nudges it to continue the authorized work. This workflow check runs
before the wire-only completion classifier, independently of the default zero
Stop-hook re-entry allowance. It shares the existing nudge and round budgets;
it cannot reset those budgets. The nudge is for an idle session only. Before
nudging, the harness drains the follow-up queue: a queued child result or an
owner message typed meanwhile becomes the next round instead of a nudge. If a
child finished in the last two minutes and its result is still being
delivered, the harness waits up to 15 seconds for it. When nothing arrives
(lookup errors included), the nudge fires as before, so a session is never
left waiting on an event that does not come. Live children, owner questions/abort, explicit
blocked replies, cancellation, budget limits, non-auto modes, disabled
continuation/protections, and memory-flush turns do not trigger that recovery.

Each `AgentAsync(runs=[...])` dispatch assigns a runtime-generated wave id to
its background children. A standalone background dispatch gets its own wave.
Wave ids are returned in job handles and in the lineage-scoped `action=list`
view, which also shows terminal/total progress. Foreground-only batches do not
create wave records.

A wave's registration barrier closes when batch dispatch settles. No member
can wake the parent before that barrier closes and all registered children are
terminal (`done`/`completed`, `failed`, `cancelled`, or `aborted`). One delivery
claim covers every member. The parent receives one digest with per-child
status, session/job result references, result text, error, and available turn
cost/token metrics. Unknown costs are explicitly `null`, not zero. Different
waves under the same parent complete and wake independently at the storage
layer; Jarvis's harness allows overlapping waves only when they belong to
different owner prompts.

`AgentAsync(action="steer", task_id=..., prompt=...)` queues same-goal guidance
to the existing child. It preserves the job and wave id and does not reset the
barrier. The caller is responsible for keeping steering within the same goal.

A wave can also be **edited mid flight** by adding a member: a dispatch (solo
or `runs=[...]` batch) that passes a started handle's `wave_id` appends its
children to that still-running wave instead of opening a new one. The barrier
extends to the appended children and the parent still wakes exactly once, with
every member — original and appended — in the same consolidated digest. Keep
this for the same goal only, exactly as with steering; a dispatch without a
`wave_id` never joins an existing wave, so different goals always run as
separate waves. Membership stays editable only while the wave is in flight:
once a delivery attempt has begun (the wave manifest's receipt key is pinned)
or any member is delivered, the wave is final and a new dispatch must open a
new wave. A `wave_id` that names no wave of this session (mistyped, made up, or
not a `wave_<32 hex>` id), another session's wave, or a final wave is refused
before any child session or job is created; the refusal says to omit `wave_id`
or dispatch a new wave. An empty or null-like value (`null`, `none`) counts as
omitted.

`AgentAsync(action="abort", wave_id=...)` cancels the whole wave and wakes the
parent immediately after cancellation bookkeeping, without waiting for engines
to finish. It also seals registration and freezes membership (the receipt key
pins immediately), so a late same-goal append is refused. Aborting one
child with the existing job/task selectors leaves the rest of that wave
running and waits for them before delivering the consolidated result. Background
jobs that an aborted child had dispatched itself are cancelled with it and
silenced: the aborted child session is never woken by its own wave. As before,
stopping the **parent** suppresses automatic continuation of that parent, and a
sibling that finished before the stop does not carry the wave back to it.

Delivery uses existing durable job-id claims and gateway queue receipts. The
receipt key is the lexically first member job id, pinned in the wave manifest
when the first delivery claim is taken, so it stays stable even if another
member or the reconciler performs delivery or a retry follows a lost POST
response. Failed POSTs release the claim; a crashed
claimant can be reclaimed. The gateway receipt prevents a duplicate parent turn
when a POST was accepted just before its caller died. Recovery ages a wave from
its latest terminal member, so a long-running sibling cannot cause an early
finisher's result to expire. An unsealed wave whose dispatcher died becomes
eligible when all registered members are terminal. Existing boot recovery
policy (no stale parent reactivation) is retained.

Records without a wave id remain independent solo waves and retain their legacy
result prompt. No job or session DB migration is needed. In-flight runners from
a previous installation therefore keep their original solo behavior.

The shared `Agent.run_turn` retry loop normally allows one attempt plus five
retries. `retry_errors.py` recognizes permanent messages even when an engine
labels them as generic `cli_error`: individual quota reached, exhausted or
exceeded quota, insufficient quota, usage-limit exhaustion, JSON-RPC `-32602`,
invalid params/parameters, `cwd must be an absolute path`, and schema-validation
errors. These become `quota_exhausted` or `invalid_params` and end the burst
immediately. Timeout, HTTP 5xx, connection reset, and ordinary HTTP 429 failures
retain the existing retry policy. User-configured retry allowlists cannot
re-enable these permanent patterns.

The WebUI waiting indicator is unchanged; its wave progress display is a
follow-up. This change requires no service operations and does not deploy itself.
