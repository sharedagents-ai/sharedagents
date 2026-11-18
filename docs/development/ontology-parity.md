# Ontology capability and evaluation plan

SharedAgents' ontology is owned by the native research agent. SharedData remains
its data layer. Every new research run should make its sources, extracted meaning,
conflicts and cited result available to later research.

The comparison baseline is NVIDIA Auto Ontology commit
`aedaf127fac95c9cb5219b64918f3f8e31959187`. Its public OpenAPI specification has 119
HTTP operations. Functional parity requires a capability-by-capability evaluation;
renaming a tool or storing JSON is not a parity test.

## Usage evidence informing the design

- [NVIDIA's documented workflow](https://github.com/NVIDIA/auto-ontology/blob/aedaf127fac95c9cb5219b64918f3f8e31959187/README.md)
  connects source data, compiles a semantic model and supports grounded questions.
- [Agent integrations](https://github.com/NVIDIA/auto-ontology/blob/aedaf127fac95c9cb5219b64918f3f8e31959187/docs/mcp.md)
  expose term inspection, readiness, answerability and querying to MCP clients.
- [Issue 497](https://github.com/NVIDIA/auto-ontology/issues/497) records a request
  for an explicit missing-data outcome. It is a reported scenario, not evidence
  that every release still has that problem.
- [Issue 499](https://github.com/NVIDIA/auto-ontology/issues/499) reports an index
  consistency failure after a scoped reset. This motivates transactional state
  and index updates plus recovery tests; issue reports are not benchmarks.
- [Issue 477](https://github.com/NVIDIA/auto-ontology/issues/477) proposes another
  storage provider, illustrating demand for separating semantic behavior from
  a particular database engine. It is a proposal, not proven adoption.

The additional SharedAgents objective is automatic, source-backed accumulation
from native research, including changing and contradictory observations. Advantage
claims require measured outcomes against a comparable baseline.

## Delivery ledger

| Capability | Current delivery | Remaining acceptance |
| --- | --- | --- |
| Native research accumulation | Native publication gate, immutable captures, definitions, claims and receipts | Broader reviewed extraction/answer quality, including role/entailment errors |
| Concurrent sources | Awaited native coordinator; transactional eight-writer storage replay | Sustained fresh-model extraction at corpus scale and measured inference concurrency |
| Evidence and time | Bounded incremental states, historical evidence paging, object relations, verified publisher corrections, separate UTC timelines | General validity intervals, ambiguous identity review, retractions and multi-hop contradiction judgments |
| Catalog and profiling | Configured SQLAlchemy sources; tables/views, keys, sample profiles and PII hints | HeavyDB adapter, live cloud/vendor matrix, comprehensive PII classification, direct vendor query-history readers |
| Semantic compilation | Native term/column refinement with coverage status; executable queries; schema families, validated inferred keys, junction discovery and exported history examples | Learned family union/reconciliation, evaluated junction suggestions and full-scale semantic quality |
| Retrieval and answering | SDK FTS5 + exact cosine kNN fusion; configured real embeddings; retrieved-schema SQL validation; archived results and chart/CSV APIs | ANN at large vector scale, nuanced answerability evaluation, chart/chat workbench and full benchmark comparison |
| Governance | Workspace memberships/roles, CAS edits, immutable review history, human certification, reviewed tagging rules and authenticated explorer | Per-object RBAC, full zones/tags/rule authoring interface and maintenance/re-embedding workflow |
| Interoperability | Python/CLI/native tools, authenticated HTTP and stdio MCP; native/NVIDIA import/export and official Ossie conversion | Complete vendor-field round trips, protocol compatibility matrix and expanded MCP acceptance |
| Predictions | Prediction definitions may be represented as drafts | Executable Kumo/PQL integration and evaluated predictions; a stored record is not this feature |
| Operations | Explicit SDK interpreter, private stores, atomic postings/vectors, stale-vector invalidation and recovery tests | Release SDK dependency first, connector deployment matrix, maintenance UI and protected GitHub delivery |

## Evaluation requirements

1. Use real research tasks as well as reviewed structured-data questions. Compare
   equivalent models and sources; report answer correctness and abstention,
   evidence quality, entity/concept quality and query execution correctness.
2. For classified-news evaluation, freeze and hash the eligible corpus, exclude
   low relevance and duplicates, record the random arrival order and retain source
   publication/event timestamps. Do not silently restart a previously stopped
   corpus job.
3. Compare randomized and chronological ingestion semantics, corrections arriving
   before originals, tied event times, syndicated reports, conflicting definitions,
   unit/period differences and unsupported conclusions.
4. Measure extraction throughput, concurrent requests, update lag, retrieval p50/p95,
   working memory, disk growth and model cost. Separate cold and warm queries.
5. Inject publication failures, retry operations, reopen read-only and check user
   isolation. Attribute every reported test to an actual command/run artifact.

Full NVIDIA parity is unfinished until the remaining entries have implemented
behavior and passing acceptance evidence. New capabilities must remain in
SharedAgents rather than adding semantic/model logic to SharedData.

## Acceptance evidence from the implementation

The acceptance commands and private run artifacts are recorded by the operator,
not bundled with deployment-specific identities, model names, timings or source
corpora in the public package. The completed checks include:

- A native research run over the public Chinook SQLite dataset compiled all 11
  tables and refined all 64 column mappings. Coverage became `draft_ready`; no
  semantic correctness certification was claimed.
- A separate native question produced the correct top billing-country revenues,
  matching direct SQLite results. Executed query and catalog revisions were pinned
  in the published receipt. This is one reviewed question, not a full benchmark.
- Replaying 4,032 frozen classified-source extractions chronologically and with
  eight concurrent shuffled workers produced identical final claim-state digests.
  Publication timelines stayed ordered and five disagreements remained contested.
  It made zero new model calls and does not represent extraction of the remaining
  eligible corpus. Exact quotes alone cannot prove semantic entailment.
- Unit/integration tests cover rollback, eight writers, stale vectors, namespace
  separation, failed publications, review permissions, inferred-key validation,
  source-schema changes, guarded SQL and native/Ossie interchange.

The currently configured SharedAgents remote is GitHub. Release/delivery must use
its protected PR workflow and scan the complete outgoing Git history and package
artifacts. An unavailable or rejected forge credential blocks publishing; it is
not a reason to publish to an older remote or claim deployment succeeded.
