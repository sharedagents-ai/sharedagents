"""Research ontology policies over a SharedData-owned transactional container.

Models propose meaning; this module validates evidence, maintains indexes and
publishes immutable research receipts. It never calls a language model.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .claims import ClaimPolicy
from .semantic import SemanticModel


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def identity(*parts):
    return hashlib.sha256(encoded(parts).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError("timestamps must be timezone-aware ISO strings")
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("timestamps must include their timezone")
    return date.astimezone(timezone.utc).isoformat()


def tokens(text):
    # Unicode words, including non-Latin business/entity names.
    return set(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE))


def canonical_url(url):
    if not isinstance(url, str) or not url.strip() or len(url) > 8192:
        raise ValueError("source URL must be a nonempty string of at most 8192 characters")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("source must be an HTTP(S) URL without embedded credentials")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


def canonical_publisher(url):
    """Separate tenants on shared publication hosts; abstain for short URLs."""
    parts = urlsplit(url)
    host, path = parts.hostname, parts.path.strip("/").split("/")
    if host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}:
        if (
            len(path) >= 3
            and path[1] == "status"
            and re.fullmatch(r"[A-Za-z0-9_]{1,15}", path[0])
            and path[0].casefold() != "i"
        ):
            return "x.com/" + path[0].casefold()
        return None
    if host in {"github.com", "raw.githubusercontent.com"}:
        return "github.com/" + "/".join(path[:2]).casefold() if len(path) >= 2 else None
    if host in {"t.co", "bit.ly", "tinyurl.com", "youtu.be", "www.youtube.com", "youtube.com", "doi.org"}:
        return None
    return host


def grounded_quote(text, quote):
    """Accept line wrapping, returning the actual contiguous source span."""
    if not isinstance(quote, str) or not quote.strip() or len(quote) > 4000:
        raise ValueError("an exact source quote of at most 4000 characters is required")
    if quote in text:
        return quote
    pattern = r"\s+".join(re.escape(part) for part in quote.split())
    match = re.search(pattern, text)
    if match:
        return match[0]
    raise ValueError("needs an exact source quote; copy source wording and Markdown punctuation")


def external_identity(value, source_url):
    """Normalize explicit identifiers, including repository IDs in repository evidence.

    No fuzzy label matching occurs. A source-document URL is not silently
    converted into its containing project's identity.
    """
    if not value:
        return None
    if not isinstance(value, str) or len(value) > 1000:
        raise ValueError("external_id must be an identifier string of at most 1000 characters")
    value = value.strip()
    source = urlsplit(source_url)
    if source.hostname == "raw.githubusercontent.com":
        parts = source.path.strip("/").split("/")
        if len(parts) >= 2:
            repository = "/".join(parts[:2]).casefold()
            probe = (
                value.removeprefix("https://")
                .removeprefix("http://")
                .removeprefix("github.com/")
                .rstrip("/")
                .removesuffix(".git")
                .casefold()
            )
            if probe == repository:
                return "https://github.com/" + repository
    if value.startswith("github.com/"):
        value = "https://" + value
    parts = urlsplit(value)
    if parts.scheme in {"http", "https"}:
        if parts.username or parts.password or not parts.hostname:
            raise ValueError("external identifiers cannot contain credentials")
        return urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, parts.fragment)
        )
    return value


class Ontology(SemanticModel, ClaimPolicy):
    """Application semantics; the supplied container owns connections and commits."""

    def __init__(self, container, *, initialize=True):
        self.container = container
        if not initialize:
            return
        with container.connection(write=True) as db:
            for sql in (
                "CREATE TABLE IF NOT EXISTS ontology_objects (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS ontology_terms (term TEXT, object_id TEXT, PRIMARY KEY(term,object_id))",
                "CREATE INDEX IF NOT EXISTS ontology_terms_object ON ontology_terms(object_id)",
                "CREATE TABLE IF NOT EXISTS ontology_runs (id TEXT PRIMARY KEY, actor TEXT NOT NULL, payload TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS ontology_sources (run_id TEXT, source_id TEXT, extracted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(run_id,source_id))",
                "CREATE TABLE IF NOT EXISTS ontology_source_observations (run_id TEXT,source_id TEXT,observed_at TEXT NOT NULL,PRIMARY KEY(run_id,source_id))",
                "CREATE TABLE IF NOT EXISTS ontology_claims (fact TEXT, claim_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, event_at TEXT)",
                "CREATE INDEX IF NOT EXISTS ontology_claim_fact ON ontology_claims(fact)",
                "CREATE INDEX IF NOT EXISTS ontology_claim_source ON ontology_claims(source_id)",
                "CREATE INDEX IF NOT EXISTS ontology_claim_time ON ontology_claims(event_at,claim_id)",
                "CREATE TABLE IF NOT EXISTS ontology_states (fact TEXT PRIMARY KEY, revision TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS ontology_definitions (object_id TEXT, definition_id TEXT, PRIMARY KEY(object_id,definition_id))",
                "CREATE TABLE IF NOT EXISTS ontology_edges (claim_id TEXT PRIMARY KEY, subject TEXT, predicate TEXT, fact TEXT)",
                "CREATE INDEX IF NOT EXISTS ontology_edge_subject ON ontology_edges(subject)",
                "CREATE INDEX IF NOT EXISTS ontology_edge_predicate ON ontology_edges(predicate)",
                "CREATE TABLE IF NOT EXISTS ontology_relation_objects (claim_id TEXT PRIMARY KEY,object_id TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS ontology_relation_object ON ontology_relation_objects(object_id)",
                "CREATE INDEX IF NOT EXISTS ontology_object_kind ON ontology_objects(kind)",
                "CREATE TABLE IF NOT EXISTS ontology_changes (sequence INTEGER PRIMARY KEY AUTOINCREMENT, revision TEXT UNIQUE NOT NULL, run_id TEXT NOT NULL, created_at TEXT NOT NULL)",
            ):
                db.execute(sql)
            self._semantic_setup(db)
            self._claims_setup(db)
            db.execute(
                "CREATE INDEX IF NOT EXISTS ontology_run_session ON ontology_runs(json_extract(payload,'$.session_id'))"
            )

    @staticmethod
    def _get(db, key):
        row = db.execute("SELECT payload FROM ontology_objects WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    @classmethod
    def _put(cls, db, key, kind, value, *, index=""):
        old = cls._get(db, key)
        if old is not None:
            if old != value:
                raise ValueError("immutable object identity reused with different content")
            return
        db.execute("INSERT INTO ontology_objects VALUES (?,?,?)", (key, kind, encoded(value)))
        db.executemany("INSERT OR IGNORE INTO ontology_terms VALUES (?,?)", [(t, key) for t in tokens(index)])

    def begin(self, run_id, *, actor, question, model, prompt_version, agent="research", session_id="", turn_id=""):
        if not all(isinstance(v, str) and v.strip() for v in (run_id, actor, question, model, prompt_version)):
            raise ValueError("run, actor, question, model and prompt version are required")
        with self.container.connection(write=True) as db:
            old = db.execute("SELECT payload FROM ontology_runs WHERE id=?", (run_id,)).fetchone()
            if old:
                run = json.loads(old[0])
                if (run["actor"], run["question"], run["model"], run["prompt_version"]) != (
                    actor,
                    question,
                    model,
                    prompt_version,
                ):
                    raise ValueError("run identity reused with different provenance")
                return run
            run = {
                "id": run_id,
                "actor": actor,
                "question": question,
                "model": model,
                "prompt_version": prompt_version,
                "started_at": now(),
                "status": "open",
                "agent": agent,
                "session_id": session_id,
                "turn_id": turn_id,
            }
            db.execute("INSERT INTO ontology_runs VALUES (?,?,?)", (run_id, actor, encoded(run)))
            return run

    @staticmethod
    def _run(db, run_id, actor, *, open_only=True):
        row = db.execute("SELECT actor,payload FROM ontology_runs WHERE id=?", (run_id,)).fetchone()
        if not row or row[0] != actor:
            raise PermissionError("research run unavailable to this actor; begin a run first")
        run = json.loads(row[1])
        if open_only and run["status"] != "open":
            raise ValueError("research run is already published")
        return run

    def check_open(self, run_id, actor):
        with self.container.connection() as db:
            return self._run(db, run_id, actor)

    def capture(
        self,
        run_id,
        actor,
        *,
        url,
        text,
        title="",
        published_at=None,
        truncated=False,
        publisher=None,
        published_at_verified=True,
        provenance=None,
    ):
        url = canonical_url(url)
        if not isinstance(text, str) or not text.strip() or len(text) > 1_000_000:
            raise ValueError("source text must contain 1..1000000 characters")
        published_at = timestamp(published_at) if published_at else None
        publisher = publisher if publisher is not None else canonical_publisher(url)
        if publisher is not None and (not isinstance(publisher, str) or not publisher or len(publisher) > 512):
            raise ValueError("publisher must be an attributed identifier or unknown")
        if not isinstance(published_at_verified, bool) or provenance is not None and not isinstance(provenance, dict):
            raise ValueError("source verification and provenance fields are malformed")
        context = {"publisher": publisher, "published_at_verified": published_at_verified, "provenance": provenance}
        key = (
            identity("source", url, text, published_at, context)
            if provenance or publisher != urlsplit(url).hostname or not published_at_verified
            else identity("source", url, text, published_at)
        )
        with self.container.connection(write=True) as db:
            self._run(db, run_id, actor)
            source = self._get(db, key)
            if source is None:
                source = {
                    "id": key,
                    "kind": "source",
                    "url": url,
                    "text": text,
                    "title": title,
                    "publisher": publisher,
                    "published_at": published_at,
                    "published_at_verified": published_at_verified,
                    "provenance": provenance,
                    "captured_at": now(),
                    "truncated": bool(truncated),
                    "content_hash": identity(text),
                }
                self._put(db, key, "source", source, index=title + " " + text)
            db.execute("INSERT OR IGNORE INTO ontology_sources(run_id,source_id) VALUES (?,?)", (run_id, key))
            db.execute("INSERT OR IGNORE INTO ontology_source_observations VALUES (?,?,?)", (run_id, key, now()))
            return source

    def extract(self, run_id, actor, source_id, *, entities, concepts, claims):
        """Publish grounded structured observations atomically, retaining disagreements.

        Evidence grounding validates exact quotes, not real-world truth. Generated
        definitions and identities remain provisional and model-attributed.
        """
        if (
            not all(isinstance(x, list) for x in (entities, concepts, claims))
            or sum(map(len, (entities, concepts, claims))) > 300
        ):
            raise ValueError("extraction needs entity/concept/claim lists totalling at most 300 items")
        semantic_index = self._model_index()
        with self.container.connection(write=True) as db:
            run = self._run(db, run_id, actor)
            if not db.execute(
                "SELECT 1 FROM ontology_sources WHERE run_id=? AND source_id=?", (run_id, source_id)
            ).fetchone():
                raise ValueError("source must have been captured in this research run")
            source = self._get(db, source_id)
            local, cardinalities = {}, {}
            objects = []
            semantic_ids = []
            for kind, items in (("entity", entities), ("concept", concepts)):
                for item in items:
                    label = str(item.get("name", "")).strip()
                    alias = str(item.get("key", "")).strip()
                    if not label or len(label) > 300 or not alias or alias in local:
                        raise ValueError(f"{kind} key {alias!r}: needs a unique local key and nonempty name")
                    try:
                        quote = grounded_quote(source["text"], item.get("quote", ""))
                    except ValueError as exc:
                        raise ValueError(f"{kind} key {alias!r}: {exc}") from exc
                    external = external_identity(item.get("external_id"), source["url"])
                    key = (
                        identity(kind, "external", external)
                        if external
                        else identity(kind, item.get("type", kind), label.casefold())
                    )
                    local[alias] = key
                    base = {
                        "id": key,
                        "kind": kind,
                        "name": external or label.casefold(),
                        "type": kind if external else str(item.get("type", kind)),
                        "external_id": external,
                        "identity_status": "provisional",
                    }
                    # A later description never replaces an existing definition.
                    if not self._get(db, key):
                        self._put(db, key, kind, base, index=label)
                    db.executemany(
                        "INSERT OR IGNORE INTO ontology_terms VALUES (?,?)", [(word, key) for word in tokens(label)]
                    )
                    proposal = {
                        "object": key,
                        "name": label,
                        "type": str(item.get("type", kind)),
                        "external_id": item.get("external_id"),
                        "source": source_id,
                        "description": str(item.get("description", "")),
                        "aliases": item.get("aliases", []),
                        "quote": quote,
                        "model": run["model"],
                        "prompt_version": run["prompt_version"],
                        "status": "proposed",
                        "cardinality": item.get("cardinality", "many" if item.get("type") == "relation" else "one"),
                    }
                    if proposal["cardinality"] not in {"one", "many"}:
                        raise ValueError("concept cardinality must be one or many")
                    cardinalities[alias] = proposal["cardinality"]
                    definition_id = identity("definition", proposal)
                    self._put(
                        db,
                        definition_id,
                        "definition",
                        proposal,
                        index=label + " " + proposal["description"] + " " + " ".join(proposal["aliases"]),
                    )
                    db.execute("INSERT OR IGNORE INTO ontology_definitions VALUES (?,?)", (key, definition_id))
                    term_key = identity("research-term", key)
                    previous_term = self._model_get(db, term_key)
                    # A stable preview is not a semantic adjudication. Retain
                    # every definition separately; do not let random arrival
                    # choose the default preview or overwrite curated meaning.
                    if previous_term is None or (
                        previous_term["origin"] == "research"
                        and previous_term["certification"]["status"] != "certified"
                        and definition_id < previous_term["data"].get("definition", definition_id)
                    ):
                        self._model_put(
                            db,
                            semantic_index,
                            "term",
                            {
                                "name": label,
                                "description": proposal["description"],
                                "aliases": proposal["aliases"],
                                "research_object": key,
                                "tables": [],
                                "definition": definition_id,
                                "definition_selection": "deterministic_provisional_preview",
                                "cardinality": proposal["cardinality"],
                            },
                            actor=actor,
                            key=term_key,
                            expected=previous_term["revision"] if previous_term else None,
                            origin="research",
                        )
                    semantic_ids.append(term_key)
                    objects.append(key)
            affected, accepted = set(), []
            for item in claims:
                subject, predicate = local.get(item.get("subject")), local.get(item.get("predicate"))
                if (
                    not subject
                    or not predicate
                    or self._get(db, subject)["kind"] not in {"entity", "concept"}
                    or self._get(db, predicate)["kind"] != "concept"
                ):
                    raise ValueError("claim subject/predicate must bind extracted node/concept keys")
                quote = grounded_quote(source["text"], item.get("quote", ""))
                if "value" not in item and "object" not in item:
                    raise ValueError("claim value is required")
                if "value" in item and "object" in item:
                    raise ValueError("claim must supply a scalar value or a node object, not both")
                event_at = timestamp(item["event_at"]) if item.get("event_at") else None
                context = {k: item.get(k) for k in ("unit", "period", "basis")}
                target = None
                value = item.get("value")
                if "object" in item:
                    target = local.get(item["object"])
                    if not target or not isinstance(item.get("polarity", True), bool):
                        raise ValueError("relation object must bind an extracted node and polarity must be boolean")
                    value = {"ref": target, "polarity": item.get("polarity", True)}
                    context["cardinality"] = cardinalities[item["predicate"]]
                    if context["cardinality"] == "many":
                        context["relation_target"] = target
                if item.get("modality"):
                    if item["modality"] not in {"actual", "planned", "expected", "possible", "rumored"}:
                        raise ValueError("unknown claim modality")
                    context["modality"] = item["modality"]
                fact = identity(subject, predicate, context)
                claim = {
                    "subject": subject,
                    "predicate": predicate,
                    "value": value,
                    "context": context,
                    "quote": quote,
                    "source": source_id,
                    "event_at": event_at,
                    "corrects": item.get("corrects"),
                    "epistemic_status": "reported",
                    "model": run["model"],
                    "prompt_version": run["prompt_version"],
                }
                if target:
                    claim["object"] = target
                key = identity("claim", claim)
                self._put(db, key, "claim", dict(claim, id=key), index=quote)
                db.execute("INSERT OR IGNORE INTO ontology_claims VALUES (?,?,?,?)", (fact, key, source_id, event_at))
                self._track_claim(db, fact, dict(claim, id=key))
                db.execute("INSERT OR IGNORE INTO ontology_edges VALUES (?,?,?,?)", (key, subject, predicate, fact))
                if target:
                    db.execute("INSERT OR IGNORE INTO ontology_relation_objects VALUES (?,?)", (key, target))
                affected.add(fact)
                accepted.append(key)
            for fact in sorted(affected):
                self._reconcile(db, fact)
            receipt = {
                "source": source_id,
                "entities": objects,
                "claims": accepted,
                "model": run["model"],
                "prompt_version": run["prompt_version"],
                "semantic_ids": sorted(set(semantic_ids)),
            }
            key = identity("extraction", receipt)
            self._put(db, key, "extraction", receipt)
            db.execute("UPDATE ontology_sources SET extracted=1 WHERE run_id=? AND source_id=?", (run_id, source_id))
            return dict(receipt=key, **receipt, facts_updated=len(affected))

    def _reconcile(self, db, fact):
        state = self._claim_snapshot(db, fact)
        revision = identity("state", state)
        self._put(db, revision, "state", state)
        db.execute(
            "INSERT OR IGNORE INTO ontology_state_evidence VALUES (?,?,?,?)",
            (
                revision,
                fact,
                db.execute("SELECT COALESCE(MAX(sequence),0) FROM ontology_claim_acceptance").fetchone()[0],
                db.execute("SELECT COALESCE(MAX(sequence),0) FROM ontology_claim_status_log").fetchone()[0],
            ),
        )
        db.execute(
            "INSERT INTO ontology_states VALUES (?,?) ON CONFLICT(fact) DO UPDATE SET revision=excluded.revision",
            (fact, revision),
        )

    def finish(self, run_id, actor, *, answer, citations, limitations):
        with self.container.connection(write=True) as db:
            run = self._run(db, run_id, actor, open_only=False)
            if run["status"] == "published":
                previous = self._get(db, run["revision"])
                if (previous["answer"], previous["citations"], previous["limitations"]) != (
                    answer,
                    sorted(set(citations)),
                    limitations,
                ):
                    raise ValueError("published answer is immutable")
                return previous
            if (
                not isinstance(answer, str)
                or not answer.strip()
                or len(answer) > 64000
                or not isinstance(limitations, list)
                or not isinstance(citations, list)
            ):
                raise ValueError("answer, citation list and limitation list are required")
            rows = db.execute("SELECT source_id,extracted FROM ontology_sources WHERE run_id=?", (run_id,)).fetchall()
            if any(not r[1] for r in rows):
                raise ValueError("all captured sources need entity/concept extraction before completion")
            referenced_sources = {r[0] for r in rows}
            referenced_queries = []
            semantic_citations = {}
            if len(citations) > 100 or not all(isinstance(c, str) for c in citations):
                raise ValueError("citations must contain at most 100 stored object IDs")
            for key in citations:
                value = db.execute("SELECT kind FROM ontology_objects WHERE id=?", (key,)).fetchone()
                if value is None:
                    semantic = self._model_get(db, key)
                    if semantic:
                        semantic_citations[key] = semantic["revision"]
                        continue
                elif value[0].startswith("semantic."):
                    semantic = self._get(db, key)
                    semantic_citations[semantic["id"]] = key
                    continue
                if not value or value[0] not in {"source", "claim", "query"}:
                    raise ValueError("citations must reference stored sources, grounded claims or executed queries")
                if value[0] == "query":
                    referenced_queries.append(key)
                    continue
                referenced_sources.add(key if value[0] == "source" else self._get(db, key)["source"])
            if rows and not citations:
                raise ValueError("a research answer must cite its stored evidence")
            if not rows and not citations and not limitations:
                raise ValueError("a run without sources must record its coverage limitation")
            states = {}
            for source_id in sorted(referenced_sources):
                states.update(
                    db.execute(
                        "SELECT s.fact,s.revision FROM ontology_claims c JOIN ontology_states s ON s.fact=c.fact WHERE c.source_id=?",
                        (source_id,),
                    )
                )
            receipt = {
                "run": run_id,
                "actor": actor,
                "question": run["question"],
                "model": run["model"],
                "prompt_version": run["prompt_version"],
                "answer": answer,
                "agent": run.get("agent", "research"),
                "session_id": run.get("session_id", ""),
                "turn_id": run.get("turn_id", ""),
                "citations": sorted(set(citations)),
                "limitations": limitations,
                "source_ids": sorted(referenced_sources),
                "captured_source_ids": sorted(r[0] for r in rows),
                "query_ids": sorted(referenced_queries),
                "semantic_citations": semantic_citations,
                "observed_at": dict(
                    db.execute(
                        "SELECT source_id,observed_at FROM ontology_source_observations WHERE run_id=?", (run_id,)
                    )
                ),
                "states": states,
            }
            revision = identity("research", receipt)
            self._put(db, revision, "research", dict(receipt, revision=revision), index=run["question"] + " " + answer)
            db.execute(
                "INSERT INTO ontology_changes(revision,run_id,created_at) VALUES (?,?,?)", (revision, run_id, now())
            )
            run.update(status="published", revision=revision)
            db.execute("UPDATE ontology_runs SET payload=? WHERE id=?", (encoded(run), run_id))
            return dict(receipt, revision=revision)

    def search(self, query, *, limit=20, kind=None):
        terms = sorted(tokens(query))[:64]
        if not terms:
            return []
        limit = max(1, min(int(limit), 100))
        with self.container.connection() as db:
            sql = (
                "SELECT o.id,COUNT(*) AS score FROM ontology_terms t JOIN ontology_objects o ON o.id=t.object_id WHERE t.term IN ("
                + ",".join("?" for _ in terms)
                + ")"
            )
            args = list(terms)
            if kind:
                sql += " AND o.kind=?"
                args.append(kind)
            sql += " GROUP BY o.id ORDER BY score DESC,o.id LIMIT ?"
            rows = db.execute(sql, (*args, limit))
            hits = []
            for row in rows:
                value = self._get(db, row[0])
                for field in ("text", "answer"):
                    if isinstance(value.get(field), str) and len(value[field]) > 1500:
                        value = dict(
                            value, **{field: value[field][:1500], field + "_length": len(value[field])}, excerpt=True
                        )
                hits.append({"object": value, "matched_terms": row[1]})
            return hits

    def get(self, key, *, offset=0, text_limit=None):
        with self.container.connection() as db:
            value = self._get(db, key)
            if value is None:
                value = self._model_get(db, key)
            if value is not None and text_limit is not None:
                offset = max(0, int(offset))
                text_limit = max(1, min(int(text_limit), 16000))
                field = "text" if "text" in value else "answer" if "answer" in value else None
                if field:
                    text = value[field]
                    value = dict(
                        value,
                        **{field: text[offset : offset + text_limit], field + "_offset": offset},
                        next_offset=offset + text_limit if len(text) > offset + text_limit else None,
                    )
            return value

    def describe(self, key, *, limit=50):
        """Bounded indexed traversal from an entity/concept to definitions and facts."""
        limit = max(1, min(int(limit), 100))
        with self.container.connection() as db:
            definitions = [
                self._get(db, r[0])
                for r in db.execute(
                    "SELECT definition_id FROM ontology_definitions WHERE object_id=? ORDER BY definition_id LIMIT ?",
                    (key, limit),
                )
            ]
            rows = db.execute(
                "SELECT revision FROM ontology_states WHERE fact IN (SELECT fact FROM ontology_edges WHERE subject=? UNION SELECT fact FROM ontology_edges WHERE predicate=? UNION SELECT e.fact FROM ontology_relation_objects r JOIN ontology_edges e ON e.claim_id=r.claim_id WHERE r.object_id=?) ORDER BY revision LIMIT ?",
                (key, key, key, limit),
            )
            return {
                "object": self._get(db, key),
                "definitions": definitions,
                "states": [self._get(db, r[0]) for r in rows],
                "limit": limit,
            }

    def status(self):
        with self.container.connection() as db:
            return {
                "ready": True,
                "storage": "SharedData.SQLite",
                "objects": dict(db.execute("SELECT kind,COUNT(*) FROM ontology_objects GROUP BY kind")),
                "research_runs": db.execute("SELECT COUNT(*) FROM ontology_runs").fetchone()[0],
                "published_runs": db.execute("SELECT COUNT(*) FROM ontology_changes").fetchone()[0],
                "facts": db.execute("SELECT COUNT(*) FROM ontology_states").fetchone()[0],
            }

    def timeline(self, *, limit=100, basis="event"):
        if basis not in {"event", "publication"}:
            raise ValueError("timeline basis must be event or publication")
        with self.container.connection() as db:
            table, column = (
                ("ontology_claims", "event_at") if basis == "event" else ("ontology_claim_data", "published_at")
            )
            rows = db.execute(
                f"SELECT claim_id,{column} FROM {table} WHERE {column} IS NOT NULL ORDER BY {column},claim_id LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            )
            groups = {}
            for key, stamp in rows:
                groups.setdefault(stamp, []).append(self._get(db, key))
            return [
                {
                    "timestamp": stamp,
                    "event_at": stamp if basis == "event" else None,
                    "published_at": stamp if basis == "publication" else None,
                    "basis": basis,
                    "claims": claims,
                    "ordering": "undetermined" if len(claims) > 1 else "single",
                }
                for stamp, claims in groups.items()
            ]

    def export(self, *, after=0, limit=100):
        with self.container.connection() as db:
            rows = db.execute(
                "SELECT sequence,revision FROM ontology_changes WHERE sequence>? ORDER BY sequence LIMIT ?",
                (int(after), max(1, min(int(limit), 100))),
            )
            return [{"sequence": row[0], "research": self._get(db, row[1])} for row in rows]

    def publications(self, session_id, *, limit=20):
        with self.container.connection() as db:
            rows = db.execute(
                "SELECT c.sequence,c.revision FROM ontology_changes c JOIN ontology_runs r ON r.id=c.run_id WHERE json_extract(r.payload,'$.session_id')=? ORDER BY c.sequence DESC LIMIT ?",
                (session_id, max(1, min(int(limit), 100))),
            )
            return [{"sequence": row[0], "research": self._get(db, row[1])} for row in rows]


def open_ontology(root: Path, scope: str, *, create=True):
    """Use SharedData's generic public SQLite primitive; no mmap or model imports."""
    try:
        from SharedData.SQLite import SQLite
    except ImportError as exc:
        raise RuntimeError("Install the SharedData SDK in this interpreter before enabling research ontology") from exc
    container = SQLite(
        "Hashs", "RT", "ONTOLOGY", "RESEARCH", user="u" + identity(scope)[:24], root=root, private=True, create=create
    )
    return Ontology(container, initialize=create)
