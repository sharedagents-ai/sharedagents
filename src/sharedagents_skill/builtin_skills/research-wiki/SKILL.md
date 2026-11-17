---
name: research-wiki
description: Build and compound a single shared, Karpathy-style research wiki at ~/wiki (sources → entities + concepts, index, log) from web research. Use for any research / external-data gathering, tradeoff analysis, deep dive, or market/company report. Owns the wiki schema, page conventions, and the DEFAULT extraction of entities and concepts from every source. The research agent ships with this skill; jarvis (the orchestrator) routes research tasks to that agent.
---

# research-wiki

Turn web research into a **persistent, compounding wiki** at `~/wiki/`. Every research
task reads the existing wiki, adds to it, and returns a cited report. The wiki is the
durable asset; the report is a view of it. **No theme subfolders** — every session writes
to the same flat folders so knowledge across topics links and compounds.

## The default: sources → entities + concepts (never skip this)

Extraction is not optional. For **every** source you ingest, you **must** by default:

1. Write/extend a **source page** (the raw, cited summary of that URL).
2. **Extract every ENTITY** it mentions — companies, people, products, exchanges,
   models, funds, protocols — and create/extend an `entities/<slug>.md` page for each,
   linking back to the source.
3. **Extract every CONCEPT** it advances — ideas, techniques, patterns, theses, metrics —
   and create/extend a `concepts/<slug>.md` page for each, linking back to the source.

A research session that wrote source pages but no entity/concept pages is **incomplete** —
the compounding value of the wiki is the entity/concept graph, not the source dump. Decompose
first, synthesize second.

## Wiki layout

```
~/wiki/
├── README.md                 # what this wiki is + how to navigate
├── index.md                  # exhaustive catalog, REGENERATED every archive step
├── log.md                    # append-only chronological ingest/query trail
├── sources/<slug>.md         # one dated summary per ingested URL (YAML frontmatter)
├── entities/<slug>.md        # one per company/person/product/exchange/model/…
├── concepts/<slug>.md        # one per idea/technique/pattern/thesis spanning entities
└── queries/<slug>.md         # filed-back research answers worth preserving
```

## Page conventions

- **Source** (`sources/<slug>.md`) — YAML frontmatter required: `title`, `url`,
  `date` (publication date), `ingested` (YYYY-MM-DD). Body: factual summary, quotes with
  figures.
- **Entity** (`entities/<slug>.md`) — factual description + a `## Sources` section linking
  every source page that mentions it. One page per real-world entity.
- **Concept** (`concepts/<slug>.md`) — definition + how it shows up across entities + a
  `## Sources` section. One page per idea/pattern/thesis that spans entities.
- **Query** (`queries/<slug>.md`) — a preserved answer to a research question.

## Wiki rules

- **Merge, never replace.** If a page exists, extend it; preserve prior consistent claims.
- **Cite everything.** Every load-bearing claim links to a source page.
- **Flag contradictions.** New facts conflicting with a page → add a `**Contradiction:**`
  callout; never silently overwrite.
- **Absolute dates only** (`YYYY-MM-DD`) — never "today"/"last week". Check the turn header.
- **Cross-reference both ways.** If page A links to B, B must acknowledge A.
- **Greedy web-research.** Finding an entity/concept already in the wiki does NOT let you
  skip fresh `WebSearch`/`WebFetch` — the existing claims may be stale.
- **Retrieval must be a real tool call.** "I checked the wiki, nothing's there" is only true
  if `scripts/check_coverage.py` (see Pipeline step 0) actually ran and its literal
  `COVERAGE_CHECK:` line is what you're reporting — not a partial `Read` plus a guessed
  conclusion, and never a fabricated count.

## Pipeline per loop (default 2 loops; caller may set N)

0. **CHECK COVERAGE** — before planning anything, run the deterministic coverage script
   with the topic's keywords/synonyms, from this skill's own directory:
   ```bash
   python3 <this-skill-dir>/scripts/check_coverage.py "<keyword1>" "<keyword2>" ...
   ```
   This greps `index.md` + every `entities/` and `concepts/` page — not a `limit`-truncated
   `Read` of the index, which only sees a slice and tells you nothing about the rest. Paste
   its exact `COVERAGE_CHECK:` output (and any matched lines beneath it) before concluding
   "new topic" or "nothing related." Never assert a specific count (e.g. "597 entities") or
   a coverage verdict the script didn't actually print — a narrated result that doesn't match
   the script's output is a hallucinated retrieval, not a real one, and the next agent will
   inherit the same false "nothing here" belief. On `WIKI_MISSING`, scaffold the wiki first
   (see Git discipline below) and re-run.
1. **PLAN** — split the query into 3–5 parallel research questions.
2. **EXECUTE** — `WebSearch` targeted, date-qualified queries (`"<entity> 2026"`,
   `"<topic> latest"`); `WebFetch` the best (primary sources preferred); synthesize.
3. **ARCHIVE** — write source pages; **extract entities + concepts into their pages**
   (the default above); regenerate `index.md`; append to `log.md`.
4. **CONSOLIDATE** — session summary: TL;DR → Abstract → ToC → body with inline citations
   → `## Sources`.
5. **FOLLOW-UP** (if not last loop) — one coherent prompt extending the investigation.

`log.md` entry format:
```
## [YYYY-MM-DD] <action> | <title>
- Sources: <paths>  · Entities: <paths>  · Concepts: <paths>  · Contradictions: <flagged>
```

## Final report

After the last loop, produce `~/wiki/reports/research_<slug>_<timestamp>/FINAL_REPORT.md`
(and surface it at `~/wiki/reports/research_<slug>_<timestamp>.md`) — **only after the wiki
(source + entity + concept pages + index + log) is updated.** Structure: TL;DR → Abstract →
Table of Contents (mandatory, in this order) → synthesis across all loops → an "Investigation
arc". Every paragraph ends with a `**Sources:**` trailer.

## Git discipline

The wiki is a Git repo. After every session:
`cd ~/wiki && git add -A && git commit -m "research: <what was ingested>"`. Multiple agents
may push — coordinate via commits, not locks. If `~/wiki` doesn't exist, scaffold it first
(the folders above + `README.md`, `index.md`, `log.md`).
