"""Incremental conflict bookkeeping with bounded immutable state summaries."""

import json


class ClaimPolicy:
    def _claims_setup(self, db):
        for statement in (
            "CREATE TABLE IF NOT EXISTS ontology_claim_data (claim_id TEXT PRIMARY KEY,fact TEXT NOT NULL,value_key TEXT NOT NULL,content_hash TEXT NOT NULL,publisher TEXT,published_at TEXT,event_at TEXT,corrects TEXT,status TEXT)",
            "CREATE INDEX IF NOT EXISTS ontology_claim_data_fact ON ontology_claim_data(fact,claim_id)",
            "CREATE INDEX IF NOT EXISTS ontology_claim_data_status ON ontology_claim_data(fact,status,claim_id)",
            "CREATE INDEX IF NOT EXISTS ontology_claim_data_value ON ontology_claim_data(fact,value_key,status,claim_id)",
            "CREATE INDEX IF NOT EXISTS ontology_claim_data_content ON ontology_claim_data(fact,content_hash,status)",
            "CREATE INDEX IF NOT EXISTS ontology_claim_data_corrects ON ontology_claim_data(corrects)",
            "CREATE INDEX IF NOT EXISTS ontology_claim_publication ON ontology_claim_data(published_at,claim_id)",
            "CREATE TABLE IF NOT EXISTS ontology_value_counts (fact TEXT,value_key TEXT,n INTEGER NOT NULL,origins INTEGER NOT NULL,PRIMARY KEY(fact,value_key))",
            "CREATE TABLE IF NOT EXISTS ontology_content_counts (fact TEXT,value_key TEXT,content_hash TEXT,n INTEGER NOT NULL,PRIMARY KEY(fact,value_key,content_hash))",
            "CREATE TABLE IF NOT EXISTS ontology_fact_counts (fact TEXT,status TEXT,n INTEGER NOT NULL,PRIMARY KEY(fact,status))",
            "CREATE TABLE IF NOT EXISTS ontology_superseded_content (fact TEXT,content_hash TEXT,PRIMARY KEY(fact,content_hash))",
            "CREATE TABLE IF NOT EXISTS ontology_migrations (name TEXT PRIMARY KEY)",
            "CREATE TABLE IF NOT EXISTS ontology_claim_acceptance (sequence INTEGER PRIMARY KEY AUTOINCREMENT,claim_id TEXT UNIQUE NOT NULL)",
            "CREATE TABLE IF NOT EXISTS ontology_claim_status_log (sequence INTEGER PRIMARY KEY AUTOINCREMENT,claim_id TEXT NOT NULL,status TEXT NOT NULL)",
            "CREATE INDEX IF NOT EXISTS ontology_claim_status_history ON ontology_claim_status_log(claim_id,sequence)",
            "CREATE TABLE IF NOT EXISTS ontology_state_evidence (revision TEXT PRIMARY KEY,fact TEXT NOT NULL,acceptance_cutoff INTEGER NOT NULL,status_cutoff INTEGER NOT NULL)",
        ):
            db.execute(statement)
        if not db.execute("SELECT 1 FROM ontology_migrations WHERE name='claim-policy-v2'").fetchone():
            for fact, key in db.execute("SELECT fact,claim_id FROM ontology_claims ORDER BY claim_id").fetchall():
                self._track_claim(db, fact, self._get(db, key), strict=False)
            db.execute("INSERT INTO ontology_migrations VALUES ('claim-policy-v2')")

    def _claim_status(self, db, key, status):
        row = dict(db.execute("SELECT * FROM ontology_claim_data WHERE claim_id=?", (key,)).fetchone())
        old = row["status"]
        if old == status:
            return
        for state, delta in ((old, -1), (status, 1)):
            if state is None:
                continue
            db.execute(
                "INSERT INTO ontology_fact_counts VALUES (?,?,?) ON CONFLICT(fact,status) DO UPDATE SET n=n+excluded.n",
                (row["fact"], state, delta),
            )
            if state == "active":
                previous = db.execute(
                    "SELECT n FROM ontology_content_counts WHERE fact=? AND value_key=? AND content_hash=?",
                    (row["fact"], row["value_key"], row["content_hash"]),
                ).fetchone()
                before = previous[0] if previous else 0
                origin_delta = int(before == 0) if delta > 0 else -int(before == 1)
                db.execute(
                    "INSERT INTO ontology_value_counts VALUES (?,?,?,?) ON CONFLICT(fact,value_key) DO UPDATE SET n=n+excluded.n,origins=origins+excluded.origins",
                    (row["fact"], row["value_key"], delta, origin_delta),
                )
                db.execute(
                    "INSERT INTO ontology_content_counts VALUES (?,?,?,?) ON CONFLICT(fact,value_key,content_hash) DO UPDATE SET n=n+excluded.n",
                    (row["fact"], row["value_key"], row["content_hash"], delta),
                )
        db.execute("UPDATE ontology_claim_data SET status=? WHERE claim_id=?", (status, key))
        db.execute("INSERT INTO ontology_claim_status_log(claim_id,status) VALUES (?,?)", (key, status))

    def _resolve_correction(self, db, key, *, strict):
        row = dict(db.execute("SELECT * FROM ontology_claim_data WHERE claim_id=?", (key,)).fetchone())
        target = db.execute("SELECT * FROM ontology_claim_data WHERE claim_id=?", (row["corrects"],)).fetchone()
        if target is None:
            if row["status"] != "superseded":
                self._claim_status(db, key, "pending")
            return
        target = dict(target)
        valid = (
            row["fact"] == target["fact"]
            and bool(row["publisher"])
            and row["publisher"] == target["publisher"]
            and row["published_at"]
            and target["published_at"]
            and row["published_at"] > target["published_at"]
            and row["content_hash"] != target["content_hash"]
        )
        if not valid:
            if strict:
                raise ValueError(
                    "correction requires an earlier same-publisher claim for the same fact and changed content"
                )
            self._claim_status(db, key, "rejected")
            return
        if row["status"] != "superseded":
            self._claim_status(db, key, "active")
        db.execute(
            "INSERT OR IGNORE INTO ontology_superseded_content VALUES (?,?)", (row["fact"], target["content_hash"])
        )
        for old in db.execute(
            "SELECT claim_id FROM ontology_claim_data WHERE fact=? AND content_hash=? AND status!='superseded'",
            (row["fact"], target["content_hash"]),
        ).fetchall():
            self._claim_status(db, old[0], "superseded")

    def _track_claim(self, db, fact, claim, *, strict=True):
        from .ontology import encoded

        key = claim["id"]
        if db.execute("SELECT 1 FROM ontology_claim_data WHERE claim_id=?", (key,)).fetchone():
            return
        source = self._get(db, claim["source"])
        db.execute("INSERT OR IGNORE INTO ontology_claim_acceptance(claim_id) VALUES (?)", (key,))
        db.execute(
            "INSERT INTO ontology_claim_data VALUES (?,?,?,?,?,?,?,?,NULL)",
            (
                key,
                fact,
                encoded(claim["value"]),
                source["content_hash"],
                source["publisher"],
                source["published_at"] if source.get("published_at_verified", False) else None,
                claim["event_at"],
                claim["corrects"],
            ),
        )
        superseded = db.execute(
            "SELECT 1 FROM ontology_superseded_content WHERE fact=? AND content_hash=?", (fact, source["content_hash"])
        ).fetchone()
        self._claim_status(db, key, "superseded" if superseded else "active")
        if claim["corrects"]:
            self._resolve_correction(db, key, strict=strict)
        # A correction may arrive before the source it corrects. Resolve just
        # those dependents; do not revisit every report for a popular fact.
        for dependent in db.execute("SELECT claim_id FROM ontology_claim_data WHERE corrects=?", (key,)).fetchall():
            self._resolve_correction(db, dependent[0], strict=False)

    def _claim_snapshot(self, db, fact):
        def examples(status=None, value=None, limit=100):
            where, args = "fact=?", [fact]
            if status:
                where += " AND status=?"
                args.append(status)
            if value is not None:
                where += " AND value_key=?"
                args.append(value)
            return [
                row[0]
                for row in db.execute(
                    "SELECT claim_id FROM ontology_claim_data WHERE " + where + " ORDER BY claim_id LIMIT ?",
                    (*args, limit),
                )
            ]

        values = db.execute(
            "SELECT value_key,n,origins FROM ontology_value_counts WHERE fact=? AND n>0 ORDER BY value_key LIMIT 64",
            (fact,),
        ).fetchall()
        value_count = db.execute("SELECT COUNT(*) FROM ontology_value_counts WHERE fact=? AND n>0", (fact,)).fetchone()[
            0
        ]
        counts = dict(db.execute("SELECT status,n FROM ontology_fact_counts WHERE fact=?", (fact,)))
        alternatives = [
            {
                "value": json.loads(value),
                "claims": examples("active", value, 16),
                "claim_count": count,
                "independent_contents": origins,
            }
            for value, count, origins in values
        ]
        total = sum(counts.values())
        return {
            "fact": fact,
            "status": "contested" if value_count > 1 else "reported" if value_count else "pending",
            "value": alternatives[0]["value"] if value_count == 1 else None,
            "alternatives": alternatives,
            "alternative_count": value_count,
            "superseded": examples("superseded"),
            "superseded_count": counts.get("superseded", 0),
            "pending": examples("pending"),
            "pending_count": counts.get("pending", 0),
            "rejected": examples("rejected"),
            "rejected_count": counts.get("rejected", 0),
            "evidence": examples(),
            "evidence_count": total,
            "examples_truncated": total > 100 or value_count > 64 or any(count > 16 for _, count, _ in values),
            "policy": "preserve-disagreement-v2",
        }

    def fact_evidence(self, fact, *, after="", limit=100, status=None, revision=None):
        with self.container.connection() as db:
            if revision:
                cutoffs = db.execute(
                    "SELECT fact,acceptance_cutoff,status_cutoff FROM ontology_state_evidence WHERE revision=?",
                    (revision,),
                ).fetchone()
                if not cutoffs or cutoffs[0] != fact:
                    legacy = self._get(db, revision)
                    if not legacy or legacy.get("fact") != fact or legacy.get("policy") != "preserve-disagreement-v1":
                        raise ValueError("historical evidence revision does not belong to this fact")
                    rows = []
                    for key in sorted(legacy["evidence"]):
                        state = (
                            "superseded"
                            if key in legacy["superseded"]
                            else "pending"
                            if key in legacy["pending"]
                            else "active"
                        )
                        if key > after and (not status or state == status):
                            rows.append({"claim": self._get(db, key), "status": state})
                    return rows[: max(1, min(int(limit), 1000))]
                query = "SELECT claim_id,state FROM (SELECT d.claim_id,(SELECT h.status FROM ontology_claim_status_log h WHERE h.claim_id=d.claim_id AND h.sequence<=? ORDER BY h.sequence DESC LIMIT 1) AS state FROM ontology_claim_data d JOIN ontology_claim_acceptance a ON a.claim_id=d.claim_id WHERE d.fact=? AND d.claim_id>? AND a.sequence<=?)"
                values = [cutoffs[2], fact, after, cutoffs[1]]
                if status:
                    query += " WHERE state=?"
                    values.append(status)
                rows = db.execute(query + " ORDER BY claim_id LIMIT ?", (*values, max(1, min(int(limit), 1000))))
                return [{"claim": self._get(db, row[0]), "status": row[1]} for row in rows]
            where, args = "fact=? AND claim_id>?", [fact, after]
            if status:
                where += " AND status=?"
                args.append(status)
            rows = db.execute(
                "SELECT claim_id,status FROM ontology_claim_data WHERE " + where + " ORDER BY claim_id LIMIT ?",
                (*args, max(1, min(int(limit), 1000))),
            )
            return [{"claim": self._get(db, row[0]), "status": row[1]} for row in rows]
