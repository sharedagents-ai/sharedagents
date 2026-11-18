"""Governed semantic models, catalog compilation and retrieval policies.

All meaning belongs to SharedAgents. Storage, postings and vector operations use
the public SharedData container/search APIs. No method calls a language model.
"""

from __future__ import annotations

import json
from collections import deque

KINDS = frozenset(
    {
        "table",
        "column",
        "term",
        "attribute",
        "relationship",
        "metric",
        "analysis",
        "tag",
        "zone",
        "rule",
        "acronym",
        "prompt",
        "family",
        "owner",
        "prediction",
    }
)


class SemanticModel:
    @staticmethod
    def _semantic_setup(db):
        db.execute(
            "CREATE TABLE IF NOT EXISTS ontology_model (id TEXT PRIMARY KEY, kind TEXT NOT NULL, revision TEXT NOT NULL, retired INTEGER NOT NULL DEFAULT 0)"
        )
        db.execute("CREATE INDEX IF NOT EXISTS ontology_model_kind ON ontology_model(kind,retired,id)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS ontology_model_audit (sequence INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT NOT NULL, revision TEXT NOT NULL, actor TEXT NOT NULL, operation TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS ontology_retrieved (run_id TEXT NOT NULL, table_id TEXT NOT NULL, PRIMARY KEY(run_id,table_id))"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS ontology_semantic_edges (id TEXT PRIMARY KEY,source TEXT NOT NULL,target TEXT NOT NULL,enabled INTEGER NOT NULL)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS ontology_semantic_edge_source ON ontology_semantic_edges(source,enabled)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS ontology_semantic_edge_target ON ontology_semantic_edges(target,enabled)"
        )
        db.execute("CREATE TABLE IF NOT EXISTS ontology_semantic_migrations (name TEXT PRIMARY KEY)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS ontology_embedding_receipts (object_id TEXT NOT NULL,model TEXT NOT NULL,revision TEXT NOT NULL,PRIMARY KEY(object_id,model))"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS ontology_semantic_names (name TEXT NOT NULL,object_id TEXT NOT NULL,PRIMARY KEY(name,object_id))"
        )
        db.execute("CREATE INDEX IF NOT EXISTS ontology_semantic_names_object ON ontology_semantic_names(object_id)")
        if not db.execute("SELECT 1 FROM ontology_semantic_migrations WHERE name='names-v1'").fetchone():
            for row in db.execute(
                "SELECT o.payload FROM ontology_model m JOIN ontology_objects o ON o.id=m.revision WHERE m.retired=0"
            ).fetchall():
                obj = json.loads(row[0])
                names = [obj["data"]["name"], *obj["data"].get("aliases", []), *obj["data"].get("synonyms", [])]
                db.executemany(
                    "INSERT OR IGNORE INTO ontology_semantic_names VALUES (?,?)",
                    [(name.strip().casefold(), obj["id"]) for name in names],
                )
            db.execute("INSERT INTO ontology_semantic_migrations VALUES ('names-v1')")
        if not db.execute("SELECT 1 FROM ontology_semantic_migrations WHERE name='edges-v1'").fetchone():
            for row in db.execute(
                "SELECT id,revision FROM ontology_model WHERE kind='relationship' AND retired=0"
            ).fetchall():
                data = json.loads(
                    db.execute("SELECT payload FROM ontology_objects WHERE id=?", (row[1],)).fetchone()[0]
                )["data"]
                db.execute(
                    "INSERT OR REPLACE INTO ontology_semantic_edges VALUES (?,?,?,?)",
                    (row[0], data["source"], data["target"], int(data.get("enabled", True))),
                )
            db.execute("INSERT INTO ontology_semantic_migrations VALUES ('edges-v1')")

    def _model_index(self, *, dimensions=None, model=None):
        try:
            from SharedData.SearchIndex import SearchIndex
        except ImportError as exc:
            raise RuntimeError("semantic search requires SharedData 6.103.0 with its search extra") from exc
        return SearchIndex(self.container, "semantic-model", dimensions=dimensions, model=model)

    def evidence_chunks(self, source_id):
        from .ontology import identity

        source = self.get(source_id)
        if not source or source.get("kind") != "source":
            raise ValueError("stored source required")
        return [
            {
                "key": identity("evidence-chunk", source_id, offset),
                "text": source["text"][offset : offset + 3000],
                "metadata": {"source_id": source_id, "offset": offset, "url": source["url"]},
            }
            for offset in range(0, len(source["text"]), 2500)
        ]

    def index_evidence(self, rows, *, model=None):
        from SharedData.SearchIndex import SearchIndex

        dimensions = len(rows[0]["vector"]) if rows and rows[0].get("vector") is not None else None
        index = SearchIndex(self.container, "research-evidence", dimensions=dimensions, model=model)
        return {"indexed": index.put(rows)}

    def evidence_search(self, query, *, vector=None, model=None, limit=20):
        from SharedData.SearchIndex import SearchIndex

        index = SearchIndex(
            self.container, "research-evidence", dimensions=len(vector) if vector else None, model=model
        )
        return index.search(query, vector=vector, limit=limit)

    def _model_get(self, db, key, *, retired=False):
        row = db.execute("SELECT revision,retired FROM ontology_model WHERE id=?", (key,)).fetchone()
        return self._get(db, row[0]) if row and (retired or not row[1]) else None

    def model_get(self, key):
        with self.container.connection() as db:
            return self._model_get(db, key)

    def model_list(self, *, kind=None, limit=100, after=""):
        limit = max(1, min(int(limit), 1000))
        with self.container.connection() as db:
            where, values = "retired=0 AND id>?", [after]
            if kind:
                where += " AND kind=?"
                values.append(kind)
            rows = db.execute(
                "SELECT revision FROM ontology_model WHERE " + where + " ORDER BY id LIMIT ?", (*values, limit)
            )
            return [self._get(db, row[0]) for row in rows]

    def _model_put(self, db, index, kind, data, *, actor, key=None, expected=None, origin="agent"):
        from .ontology import encoded, identity, now

        if kind not in KINDS or not isinstance(data, dict) or not actor:
            raise ValueError("known semantic kind, object data and authenticated actor are required")
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            raise ValueError("semantic objects require a name")
        if len(data["name"]) > 300 or not isinstance(data.get("description", ""), str):
            raise ValueError("semantic name is too long or description is not text")
        for field in ("aliases", "synonyms"):
            values = data.get(field, [])
            if (
                not isinstance(values, list)
                or len(values) > 100
                or not all(isinstance(value, str) and len(value) <= 300 for value in values)
            ):
                raise ValueError(field + " must contain at most 100 short strings")
        if len(encoded(data)) > 128000:
            raise ValueError("semantic object exceeds the size limit")
        if set(data) & {"certification", "revision", "actor", "origin", "retired"}:
            raise ValueError("provenance and certification are controlled fields")
        key = key or identity(
            "semantic",
            kind,
            data.get("connection"),
            data.get("schema"),
            [data.get(field) for field in ("term", "column", "table", "source", "target", "tables")],
            data["name"],
        )
        old = self._model_get(db, key, retired=True)
        if old and old["kind"] != kind:
            raise ValueError("semantic identity cannot change kind")
        if old and kind in {"table", "column"} and data["name"] != old["data"]["name"] and origin != "compiler":
            raise ValueError("catalog identifiers belong to the source; edit business term names instead")
        if old and old["revision"] != expected:
            raise ValueError("semantic revision conflict; read the current revision before editing")
        if old and kind in {"table", "column"} and origin == "human":
            annotations = {"description", "aliases", "synonyms", "owner", "tags", "zones"}
            if {k: v for k, v in data.items() if k not in annotations} != {
                k: v for k, v in old["data"].items() if k not in annotations
            }:
                raise ValueError("physical catalog fields belong to the configured source")
        if (
            old
            and old["certification"]["status"] == "certified"
            and origin == "compiler"
            and kind not in {"table", "column", "relationship"}
        ):
            return old
        if kind == "term":
            tables = data.get("tables", [])
            if (
                not isinstance(tables, list)
                or len(tables) > 100
                or not all(
                    isinstance(key, str) and (self._model_get(db, key) or {}).get("kind") == "table" for key in tables
                )
            ):
                raise ValueError("term mappings must reference stored catalog tables")
        if kind == "relationship":
            if origin == "agent" and data.get("enabled", True):
                raise ValueError("agent-proposed joins remain disabled until source validation or human review")
            if not isinstance(data.get("enabled", True), bool):
                raise ValueError("relationship enabled must be boolean")
            for field in ("source", "target"):
                if not isinstance(data.get(field), str) or self._model_get(db, data[field]) is None:
                    raise ValueError("relationship endpoints must reference stored semantic objects")
        for field in ("term", "column", "source", "target"):
            if kind in {"attribute", "relationship", "metric"} and data.get(field):
                target = self._model_get(db, data[field])
                if target is None:
                    raise ValueError("semantic reference does not exist: " + field)
        certification = {"status": "draft"}
        if old and data == old["data"] and not old.get("retired"):
            return old
        body = {
            "id": key,
            "kind": kind,
            "data": data,
            "actor": actor,
            "origin": origin,
            "certification": certification,
            "previous": old["revision"] if old else None,
            "retired": False,
        }
        revision = identity("semantic-revision", body)
        body["revision"] = revision
        self._put(db, revision, "semantic." + kind, body)
        db.execute(
            "INSERT INTO ontology_model VALUES (?,?,?,0) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,retired=0",
            (key, kind, revision),
        )
        db.execute("DELETE FROM ontology_semantic_names WHERE object_id=?", (key,))
        names = [data["name"], *data.get("aliases", []), *data.get("synonyms", [])]
        db.executemany(
            "INSERT OR IGNORE INTO ontology_semantic_names VALUES (?,?)",
            [(name.strip().casefold(), key) for name in names],
        )
        if kind == "relationship":
            db.execute(
                "INSERT INTO ontology_semantic_edges VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET source=excluded.source,target=excluded.target,enabled=excluded.enabled",
                (key, data["source"], data["target"], int(data.get("enabled", True))),
            )
        db.execute(
            "INSERT INTO ontology_model_audit(object_id,revision,actor,operation,created_at) VALUES (?,?,?,?,?)",
            (key, revision, actor, "edit" if old else "create", now()),
        )
        text = " ".join(str(data.get(field, "")) for field in ("name", "description", "aliases", "synonyms", "sql"))
        index.put([{"key": key, "text": text, "metadata": {"kind": kind}}], connection=db)
        return body

    def model_put(self, kind, data, *, actor, key=None, expected=None, origin="agent"):
        if origin == "agent" and kind in {"table", "column"}:
            raise ValueError("catalog snapshots come from configured sources; define term or attribute meaning instead")
        index = self._model_index()
        with self.container.connection(write=True) as db:
            return self._model_put(db, index, kind, data, actor=actor, key=key, expected=expected, origin=origin)

    def model_put_many(self, rows, *, actor):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            raise ValueError("submit 1..32 semantic proposals per atomic batch")
        index = self._model_index()
        with self.container.connection(write=True) as db:
            results = []
            for row in rows:
                if row.get("kind") in {"table", "column"}:
                    raise ValueError("catalog snapshots cannot be authored by an agent")
                results.append(
                    self._model_put(
                        db,
                        index,
                        row["kind"],
                        row["data"],
                        actor=actor,
                        key=row.get("id"),
                        expected=row.get("expected_revision"),
                    )
                )
            return {"objects": results, "count": len(results)}

    def table_context(self, key, *, offset=0, limit=16):
        from .ontology import identity

        table = self.model_get(key)
        if not table or table["kind"] != "table":
            raise ValueError("stored catalog table required")
        offset, limit = max(0, int(offset)), max(1, min(int(limit), 32))
        columns = table["data"]["columns"]
        profiles = []
        for column in columns[offset : offset + limit]:
            column_id = identity("column", key, column["name"])
            record = self.model_get(column_id)
            if record:
                data = dict(record["data"])
                if "sample_values" in data:
                    data["sample_values"] = data["sample_values"][:5]
                profiles.append(dict(record, data=data))
        data = {field: value for field, value in table["data"].items() if field != "columns"}
        return {
            "table": dict(table, data=data),
            "term": self.model_get(identity("table-term", key)),
            "columns": profiles,
            "column_count": len(columns),
            "next_offset": offset + limit if offset + limit < len(columns) else None,
            "samples_are_previews": True,
        }

    def retrieve_table(self, run_id, actor, key, *, offset=0, limit=16):
        context = self.table_context(key, offset=offset, limit=limit)
        with self.container.connection(write=True) as db:
            self._run(db, run_id, actor)
            db.execute("INSERT OR IGNORE INTO ontology_retrieved VALUES (?,?)", (run_id, key))
        return context

    def compilation_status(self, connection):
        from .ontology import identity

        with self.container.connection() as db:
            tables = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT o.payload FROM ontology_model m JOIN ontology_objects o ON o.id=m.revision WHERE m.kind='table' AND m.retired=0 AND json_extract(o.payload,'$.data.connection')=? ORDER BY m.id",
                    (connection,),
                )
            ]
            attributes = {}
            for row in db.execute(
                "SELECT o.payload FROM ontology_model m JOIN ontology_objects o ON o.id=m.revision WHERE m.kind='attribute' AND m.retired=0"
            ):
                obj = json.loads(row[0])
                if len(obj["data"].get("description", "").strip()) >= 20:
                    attributes.setdefault(obj["data"].get("term"), set()).add(obj["data"].get("column"))
            missing, ready = [], 0
            for table in tables:
                term = self._model_get(db, identity("table-term", table["id"]))
                mapped = attributes.get(term["id"], set()) if term else set()
                absent = [
                    identity("column", table["id"], col["name"])
                    for col in table["data"]["columns"]
                    if identity("column", table["id"], col["name"]) not in mapped
                ]
                description = term["data"].get("description", "") if term else ""
                term_ready = len(description.strip()) >= 20 and not description.startswith("Draft from catalog")
                if absent or not term_ready:
                    missing.append(
                        {
                            "table": table["id"],
                            "name": table["data"]["name"],
                            "term": term["id"] if term else None,
                            "term_description_required": not term_ready,
                            "missing_columns": absent,
                        }
                    )
                else:
                    ready += 1
            return {
                "connection": connection,
                "status": "draft_ready" if tables and not missing else "refinement_required",
                "tables": len(tables),
                "ready_tables": ready,
                "missing": missing,
                "semantic_correctness_certified": False,
            }

    def model_review(self, key, *, actor, expected, certified, reason=""):
        """Explicit reviewer action; not exposed as an agent mutation capability."""
        from .ontology import identity, now

        if not actor or not isinstance(certified, bool):
            raise ValueError("reviewer and boolean certification are required")
        with self.container.connection(write=True) as db:
            old = self._model_get(db, key)
            if not old or old["revision"] != expected:
                raise ValueError("semantic revision conflict")
            body = dict(
                old,
                certification={"status": "certified" if certified else "draft", "reviewer": actor, "reason": reason},
                previous=expected,
            )
            body.pop("revision")
            body["revision"] = identity("semantic-revision", body)
            self._put(db, body["revision"], "semantic." + body["kind"], body)
            db.execute("UPDATE ontology_model SET revision=? WHERE id=?", (body["revision"], key))
            db.execute(
                "INSERT INTO ontology_model_audit(object_id,revision,actor,operation,created_at) VALUES (?,?,?,?,?)",
                (key, body["revision"], actor, "review", now()),
            )
            return body

    def model_retire(self, key, *, actor, expected):
        from .ontology import identity, now

        index = self._model_index()
        with self.container.connection(write=True) as db:
            old = self._model_get(db, key)
            if not actor or not old or old["revision"] != expected:
                raise ValueError("semantic revision conflict")
            if old["certification"]["status"] == "certified":
                raise ValueError("remove certification explicitly before retiring a reviewed object")
            body = dict(old, retired=True, previous=expected, actor=actor)
            body.pop("revision")
            body["revision"] = identity("semantic-revision", body)
            self._put(db, body["revision"], "semantic." + body["kind"], body)
            db.execute("UPDATE ontology_model SET retired=1,revision=? WHERE id=?", (body["revision"], key))
            db.execute("UPDATE ontology_semantic_edges SET enabled=0 WHERE id=?", (key,))
            index.delete([key], connection=db)
            db.execute(
                "INSERT INTO ontology_model_audit(object_id,revision,actor,operation,created_at) VALUES (?,?,?,?,?)",
                (key, body["revision"], actor, "retire", now()),
            )
            return body

    def model_history(self, *, key=None, after=0, limit=100):
        with self.container.connection() as db:
            rows = db.execute(
                "SELECT sequence,object_id,revision,actor,operation,created_at FROM ontology_model_audit WHERE sequence>?"
                + (" AND object_id=?" if key else "")
                + " ORDER BY sequence LIMIT ?",
                (int(after), *((key,) if key else ()), max(1, min(int(limit), 1000))),
            )
            return [dict(row) for row in rows]

    def model_search(self, query="", *, vector=None, model=None, kind=None, limit=20):
        index = self._model_index(dimensions=len(vector) if vector else None, model=model)
        rows = index.search(query, vector=vector, limit=limit, metadata={"kind": kind} if kind else None)
        with self.container.connection() as db:
            return [{"object": self._model_get(db, row["key"]), "score": row["score"]} for row in rows]

    def model_embed(self, rows, *, model):
        if not rows:
            return {"indexed": 0}
        index = self._model_index(dimensions=len(rows[0]["vector"]), model=model)
        with self.container.connection(write=True) as db:
            prepared = []
            for row in rows:
                obj = self._model_get(db, row["id"])
                if not obj or obj["revision"] != row["revision"]:
                    raise ValueError("semantic object changed during embedding; retry current revision")
                data = obj["data"]
                text = " ".join(
                    str(data.get(field, "")) for field in ("name", "description", "aliases", "synonyms", "sql")
                )
                prepared.append(
                    {"key": obj["id"], "text": text, "vector": row["vector"], "metadata": {"kind": obj["kind"]}}
                )
            indexed = index.put(prepared, connection=db)
            db.executemany(
                "INSERT INTO ontology_embedding_receipts VALUES (?,?,?) ON CONFLICT(object_id,model) DO UPDATE SET revision=excluded.revision",
                [(row["id"], model, row["revision"]) for row in rows],
            )
            return {"indexed": indexed}

    def embedding_candidates(self, keys, *, model):
        """Return only current revisions without a committed embedding receipt."""
        if not isinstance(keys, list) or len(keys) > 500:
            raise ValueError("provide at most 500 semantic object identifiers")
        with self.container.connection() as db:
            pending = []
            for key in dict.fromkeys(keys):
                obj = self._model_get(db, key)
                receipt = db.execute(
                    "SELECT revision FROM ontology_embedding_receipts WHERE object_id=? AND model=?", (key, model)
                ).fetchone()
                if obj and (not receipt or receipt[0] != obj["revision"]):
                    pending.append(obj)
            return pending

    def apply_rules(self, *, actor):
        """Apply reviewed literal tagging rules to drafts; preserve reviewed objects."""
        rules = [
            row
            for row in self.model_list(kind="rule", limit=1000)
            if row["certification"]["status"] == "certified" and row["data"].get("enabled", True)
        ]
        changed, deferred, after = [], [], ""
        index = self._model_index()
        while True:
            rows = self.model_list(limit=1000, after=after)
            if not rows:
                break
            after = rows[-1]["id"]
            with self.container.connection(write=True) as db:
                for obj in rows:
                    if obj["kind"] in {"rule", "tag", "zone", "owner"}:
                        continue
                    data = dict(obj["data"])
                    for rule in rules:
                        predicate = rule["data"]
                        if predicate.get("target_kind") not in (None, obj["kind"]):
                            continue
                        field = predicate.get("field", "name")
                        needle = predicate.get("contains")
                        if field not in {"name", "description"} or not isinstance(needle, str) or not needle:
                            continue
                        if needle.casefold() in str(data.get(field, "")).casefold():
                            for key, kind in (("tags", "tag"), ("zones", "zone")):
                                assigned = set(data.get(key, []))
                                for identifier in predicate.get(key, []):
                                    if (self._model_get(db, identifier) or {}).get("kind") != kind:
                                        raise ValueError("rule references an unavailable " + kind)
                                    assigned.add(identifier)
                                if assigned:
                                    data[key] = sorted(assigned)
                    if data == obj["data"]:
                        continue
                    current = self._model_get(db, obj["id"])
                    if current["certification"]["status"] == "certified" or current["revision"] != obj["revision"]:
                        deferred.append(obj["id"])
                        continue
                    changed.append(
                        self._model_put(
                            db,
                            index,
                            obj["kind"],
                            data,
                            actor=actor,
                            key=obj["id"],
                            expected=obj["revision"],
                            origin="rules",
                        )["id"]
                    )
            if len(rows) < 1000:
                break
        return {"changed": changed, "review_required": deferred}

    def model_analytics(self):
        with self.container.connection() as db:
            return {
                "objects": dict(db.execute("SELECT kind,COUNT(*) FROM ontology_model WHERE retired=0 GROUP BY kind")),
                "reviews": db.execute("SELECT COUNT(*) FROM ontology_model_audit WHERE operation='review'").fetchone()[
                    0
                ],
                "published_research": db.execute("SELECT COUNT(*) FROM ontology_changes").fetchone()[0],
                "queries": db.execute("SELECT COUNT(*) FROM ontology_objects WHERE kind='query'").fetchone()[0],
                "revisions": db.execute("SELECT COUNT(*) FROM ontology_model_audit").fetchone()[0],
            }

    def model_path(self, source, target, *, max_depth=6):
        queue, visited = deque([(source, [])]), {source}
        with self.container.connection() as db:
            while queue and len(visited) <= 4096:
                key, path = queue.popleft()
                if key == target:
                    return path
                if len(path) >= min(int(max_depth), 12):
                    continue
                edges = db.execute(
                    "SELECT id,source,target FROM ontology_semantic_edges WHERE enabled=1 AND (source=? OR target=?) ORDER BY id LIMIT 500",
                    (key, key),
                ).fetchall()
                for edge_id, first, second in edges:
                    neighbor = second if first == key else first
                    if neighbor not in visited and self._model_get(db, neighbor) is not None:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [self._model_get(db, edge_id)]))
        return []

    def model_retrieve(self, run_id, actor, query, *, limit=20, vector=None, model=None):
        hits = self.model_search(query, vector=vector, model=model, limit=limit)
        tables = set()
        with self.container.connection(write=True) as db:
            self._run(db, run_id, actor)
            for hit in hits:
                obj = hit["object"]
                if obj["kind"] == "table":
                    tables.add(obj["id"])
                tables.update(obj["data"].get("tables", []))
                if obj["data"].get("table"):
                    tables.add(obj["data"]["table"])
            tables = {key for key in tables if (self._model_get(db, key) or {}).get("kind") == "table"}
            db.executemany("INSERT OR IGNORE INTO ontology_retrieved VALUES (?,?)", [(run_id, key) for key in tables])
            catalog = [self._model_get(db, key) for key in sorted(tables)]
        from itertools import combinations, islice

        pairs = list(islice(combinations(sorted(tables), 2), 64))
        paths = [self.model_path(first, second) for first, second in pairs]
        return {
            "matches": hits,
            "tables": catalog,
            "join_paths": [path for path in paths if path],
            "join_paths_truncated": len(tables) * (len(tables) - 1) // 2 > len(pairs),
        }

    def model_coverage(self, required):
        if (
            not isinstance(required, list)
            or not required
            or len(required) > 64
            or not all(isinstance(x, str) and x.strip() for x in required)
        ):
            raise ValueError("provide 1..64 required business concepts")
        available, missing = {}, []
        with self.container.connection() as db:
            for term in required:
                exact = [
                    row[0]
                    for row in db.execute(
                        "SELECT n.object_id FROM ontology_semantic_names n JOIN ontology_model m ON m.id=n.object_id WHERE n.name=? AND m.retired=0 ORDER BY n.object_id",
                        (term.strip().casefold(),),
                    )
                ]
                if exact:
                    available[term] = exact
                else:
                    missing.append(term)
        return {
            "status": "covered" if not missing else "missing_data",
            "coverage": len(available) / len(required),
            "matched": available,
            "missing": missing,
            "semantic_correctness_verified": False,
        }

    def retrieved_tables(self, run_id, actor):
        with self.container.connection() as db:
            self._run(db, run_id, actor)
            return [row[0] for row in db.execute("SELECT table_id FROM ontology_retrieved WHERE run_id=?", (run_id,))]

    def query_catalog(self, run_id, actor, connection):
        with self.container.connection() as db:
            self._run(db, run_id, actor)
            rows = db.execute(
                "SELECT o.payload FROM ontology_retrieved r JOIN ontology_model m ON m.id=r.table_id JOIN ontology_objects o ON o.id=m.revision WHERE r.run_id=? AND m.retired=0 AND m.kind='table' AND json_extract(o.payload,'$.data.connection')=? ORDER BY m.id",
                (run_id, connection),
            )
            return [json.loads(row[0]) for row in rows]

    def record_query(self, run_id, actor, *, connection, sql, result, tables):
        from .ontology import identity, now

        with self.container.connection(write=True) as db:
            self._run(db, run_id, actor)
            revisions = {}
            for key in tables:
                table = self._model_get(db, key)
                if table is None:
                    raise ValueError("query catalog changed; retrieve and retry")
                revisions[key] = table["revision"]
            payload = {
                "run": run_id,
                "actor": actor,
                "connection": connection,
                "sql": sql,
                "result": result,
                "catalog_revisions": revisions,
            }
            key = identity("query", payload)
            old = self._get(db, key)
            if old:
                return old
            payload.update(id=key, kind="query", executed_at=now())
            self._put(db, key, "query", payload, index=sql)
            return payload

    def model_export(self):
        objects, after = [], ""
        while True:
            batch = self.model_list(limit=1000, after=after)
            objects.extend(batch)
            if len(batch) < 1000:
                break
            after = batch[-1]["id"]
        return {"format": "sharedagents-semantic-v1", "objects": objects}

    def model_import(self, document, *, actor):
        if (
            not isinstance(document, dict)
            or document.get("format") != "sharedagents-semantic-v1"
            or not isinstance(document.get("objects"), list)
        ):
            raise ValueError("unsupported semantic document; import never treats unknown formats as an empty model")
        index = self._model_index()
        with self.container.connection(write=True) as db:
            staged = document["objects"]
            if len(staged) > 10000:
                raise ValueError("import exceeds 10000 objects")
            # Parent objects first. Imported certification never grants trust.
            order = {"table": 0, "column": 1, "term": 2, "attribute": 3, "relationship": 4, "metric": 5}
            results = []
            for row in sorted(staged, key=lambda obj: order.get(obj["kind"], 6)):
                old = self._model_get(db, row["id"])
                results.append(
                    self._model_put(
                        db,
                        index,
                        row["kind"],
                        row["data"],
                        actor=actor,
                        key=row["id"],
                        expected=old["revision"] if old else None,
                        origin="import",
                    )
                )
            return {"imported": len(results), "ids": [row["id"] for row in results]}

    def compile_catalog(self, connection, catalog, *, actor, analysis=None, query_history=None):
        """Compile a deterministic draft; native research supplies semantic refinement."""
        from .ontology import identity

        index = self._model_index()
        with self.container.connection(write=True) as db:
            output, table_ids = [], {}
            for table in catalog:
                key = identity("catalog", connection, table.get("schema", ""), table["name"])
                table_ids[(table.get("schema", ""), table["name"])] = key
                old = self._model_get(db, key)
                data = dict(table, connection=connection)
                if old:
                    data.update(
                        {
                            field: old["data"][field]
                            for field in ("description", "aliases", "synonyms", "owner", "tags", "zones")
                            if field in old["data"]
                        }
                    )
                output.append(
                    self._model_put(
                        db,
                        index,
                        "table",
                        data,
                        actor=actor,
                        key=key,
                        expected=old["revision"] if old else None,
                        origin="compiler",
                    )
                )
                for column in table["columns"]:
                    column_key = identity("column", key, column["name"])
                    old_column = self._model_get(db, column_key)
                    column_data = dict(column, table=key, connection=connection)
                    if old_column:
                        column_data.update(
                            {
                                field: old_column["data"][field]
                                for field in ("description", "aliases", "synonyms", "owner", "tags", "zones")
                                if field in old_column["data"]
                            }
                        )
                    output.append(
                        self._model_put(
                            db,
                            index,
                            "column",
                            column_data,
                            actor=actor,
                            key=column_key,
                            expected=old_column["revision"] if old_column else None,
                            origin="compiler",
                        )
                    )
                term_key = identity("table-term", key)
                existing = self._model_get(db, term_key)
                if not existing:
                    output.append(
                        self._model_put(
                            db,
                            index,
                            "term",
                            {
                                "name": table["name"].replace("_", " "),
                                "description": "Draft from catalog; native research review required",
                                "tables": [key],
                                "connection": connection,
                            },
                            actor=actor,
                            key=term_key,
                            origin="compiler",
                        )
                    )
            for table in catalog:
                source = table_ids[(table.get("schema", ""), table["name"])]
                for fk in table.get("foreign_keys", []):
                    target = table_ids.get((fk.get("schema") or table.get("schema", ""), fk["table"]))
                    if target:
                        data = {
                            "name": table["name"] + " -> " + fk["table"],
                            "source": source,
                            "target": target,
                            "columns": fk["columns"],
                            "target_columns": fk["target_columns"],
                            "declared": True,
                            "enabled": True,
                            "connection": connection,
                            "catalog_revisions": {
                                key: self._model_get(db, key)["revision"] for key in (source, target)
                            },
                        }
                        key = identity("declared-relationship", source, target, fk["columns"])
                        old = self._model_get(db, key)
                        output.append(
                            self._model_put(
                                db,
                                index,
                                "relationship",
                                data,
                                actor=actor,
                                key=key,
                                expected=old["revision"] if old else None,
                                origin="compiler",
                            )
                        )
            current_relationships = {obj["id"] for obj in output if obj["kind"] == "relationship"}
            for inferred in (analysis or {}).get("relationships", []):
                source = table_ids.get(tuple(inferred["source_table"]))
                target = table_ids.get(tuple(inferred["target_table"]))
                if not source or not target:
                    continue
                data = {k: v for k, v in inferred.items() if k not in {"source_table", "target_table"}}
                data.update(
                    name=inferred["source_table"][1] + " -> " + inferred["target_table"][1],
                    source=source,
                    target=target,
                    connection=connection,
                    catalog_revisions={key: self._model_get(db, key)["revision"] for key in (source, target)},
                )
                key = identity("inferred-relationship", source, target, data["columns"])
                old = self._model_get(db, key)
                current_relationships.add(key)
                output.append(
                    self._model_put(
                        db,
                        index,
                        "relationship",
                        data,
                        actor=actor,
                        key=key,
                        expected=old["revision"] if old else None,
                        origin="compiler",
                    )
                )
            # A removed declaration or a changed source snapshot invalidates
            # old join validation. Keep its history and disable traversal.
            for row in db.execute(
                "SELECT revision FROM ontology_model WHERE kind='relationship' AND retired=0"
            ).fetchall():
                old = self._get(db, row[0])
                data = old["data"]
                if (
                    data.get("connection") != connection
                    or old["origin"] != "compiler"
                    or old["id"] in current_relationships
                ):
                    continue
                pinned = data.get("catalog_revisions", {})
                changed = any(
                    self._model_get(db, key) is None or self._model_get(db, key)["revision"] != revision
                    for key, revision in pinned.items()
                )
                if data.get("declared") or changed:
                    output.append(
                        self._model_put(
                            db,
                            index,
                            "relationship",
                            dict(data, enabled=False, validation={"status": "stale_source"}),
                            actor=actor,
                            key=old["id"],
                            expected=old["revision"],
                            origin="compiler",
                        )
                    )
            for members in (analysis or {}).get("families", []):
                tables = sorted(table_ids[tuple(item)] for item in members)
                key = identity("schema-family", connection, tables)
                old = self._model_get(db, key)
                output.append(
                    self._model_put(
                        db,
                        index,
                        "family",
                        {
                            "name": " / ".join(item[1] for item in members),
                            "tables": tables,
                            "connection": connection,
                            "description": "Tables with the same observed column/type signature; no union is assumed.",
                        },
                        actor=actor,
                        key=key,
                        expected=old["revision"] if old else None,
                        origin="compiler",
                    )
                )
            for row in query_history or []:
                from .connectors import validate_sql

                tables = [self._model_get(db, key) for key in table_ids.values()]
                try:
                    validated = validate_sql(
                        row["sql"], tables, dialect=row.get("dialect", "sqlite"), retrieved=set(table_ids.values())
                    )
                except ValueError:
                    # Invalid historical queries are never promoted to examples.
                    continue
                data = {
                    "name": row.get("name", "Historical query " + identity(row["sql"])[:10]),
                    "description": row.get(
                        "description",
                        "A validated historical query; execution and business meaning require native research review.",
                    ),
                    "sql": validated["sql"],
                    "tables": validated["tables"],
                    "connection": connection,
                    "executed": False,
                }
                key = identity("query-history", connection, validated["sql"])
                old = self._model_get(db, key)
                output.append(
                    self._model_put(
                        db,
                        index,
                        "analysis",
                        data,
                        actor=actor,
                        key=key,
                        expected=old["revision"] if old else None,
                        origin="compiler",
                    )
                )
        applied = self.apply_rules(actor=actor)
        for key in applied["changed"]:
            output.append(self.model_get(key))
        output = [self.model_get(key) for key in dict.fromkeys(obj["id"] for obj in output)]
        return {
            "connection": connection,
            "objects": output,
            "schema_analysis": analysis,
            "semantic_refinement_required": True,
        }
