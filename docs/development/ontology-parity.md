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
| Native research accumulation | Source capture, structured proposals, publication gate, indexed retrieval and receipts | Broader research-quality evaluation |
| Concurrent sources | Transactional writers and awaited native worker coordinator | Measure real model concurrency and cost at scale |
| Evidence and time | Provisional meaning, quoted claims, conflict retention, explicit corrections, event ordering | Cardinality and validity intervals; reviewed ambiguous identity, retraction and multi-hop contradiction suites |
| Semantic compilation | Application/storage boundary | Ten connectors, schema profiling, query history, PII, table families, inferred joins and executable measures |
| Retrieval and answering | Native research reasoning and lexical evidence access | Semantic/name hybrid retrieval, answerability grading, validated SQL, conversational analysis and charts |
| Governance | Immutable proposals and published receipts | Certification, ownership, fine-grained ontology permissions, rules, tags, zones and management UI |
| Interoperability | Existing native agent interfaces | Complete Python/CLI/REST/MCP contracts and native/Ossie round trips |
| Predictions | Existing domain integration remains separate | Optional Kumo/PQL parity and evaluated predictions |
| Operations | Lazy SDK dependency, private stores and atomic operations | Connector deployment matrix, maintenance/recovery UI and full parity suite |

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
