"""Semantic acceptance against real SharedData transactions and live local SQL."""

import sqlite3

import pytest

from sharedagents_research.connectors import Source, validate_sql
from sharedagents_research.ontology import open_ontology


@pytest.fixture
def source(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    path = tmp_path / "sales.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, email TEXT);"
            "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), amount REAL);"
            "INSERT INTO customers VALUES(1,'Acme','contact@example.com');"
            "INSERT INTO orders VALUES(1,1,100),(2,1,50);"
        )
    monkeypatch.setenv("ONTOLOGY_TEST_SOURCE_URL", "sqlite:///" + str(path))
    result = Source("sales", scope="operator", settings={"url_env": "ONTOLOGY_TEST_SOURCE_URL"})
    yield result
    result.close()


@pytest.fixture
def model(tmp_path):
    pytest.importorskip("SharedData.SearchIndex")
    return open_ontology(tmp_path / "ontology", "test-user")


def test_catalog_terms_joins_profiling_and_validated_answer(model, source):
    result = model.compile_catalog("sales", source.catalog(), actor="test-user")
    assert result["semantic_refinement_required"]
    tables = model.model_list(kind="table")
    assert len(tables) == 2
    customer = next(t for t in tables if t["data"]["name"] == "customers")
    orders = next(t for t in tables if t["data"]["name"] == "orders")
    email = next(c for c in customer["data"]["columns"] if c["name"] == "email")
    assert email["pii_suspected"] and email["sample_values"] == []
    assert model.model_path(customer["id"], orders["id"])[0]["data"]["declared"]
    model.begin("run", actor="test-user", question="Revenue by customer", model="test-model", prompt_version="test")
    context = model.model_retrieve("run", "test-user", "orders customers")
    assert len(context["tables"]) == 2
    retrieved = {t["id"] for t in context["tables"]}
    validated = validate_sql(
        "SELECT c.name, SUM(o.amount) AS revenue FROM orders o JOIN customers c ON o.customer_id=c.id GROUP BY c.name",
        tables,
        dialect="sqlite",
        retrieved=retrieved,
    )
    assert source.run(validated["sql"])["rows"] == [["Acme", 150.0]]
    assert model.model_coverage(["orders", "ad spend"])["missing"] == ["ad spend"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "SELECT * INTO copied FROM orders",
        "SELECT * FROM orders; DROP TABLE orders",
        "SELECT unknown_column FROM orders",
        "SELECT * FROM missing",
        "SELECT load_extension('extension') FROM orders",
    ],
)
def test_query_validation_rejects_write_unknown_schema_and_function(model, source, sql):
    pytest.importorskip("sqlglot")
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    tables = model.model_list(kind="table")
    with pytest.raises(ValueError):
        validate_sql(sql, tables, dialect="sqlite", retrieved={t["id"] for t in tables})
    assert source.run("SELECT COUNT(*) FROM orders")["rows"] == [[2]]


def test_unknown_retrieval_is_not_sql_authorization(model, source):
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    with pytest.raises(ValueError, match="retrieve"):
        validate_sql(
            "SELECT SUM(amount) FROM orders", model.model_list(kind="table"), dialect="sqlite", retrieved=set()
        )


def test_explicit_table_inspection_counts_as_retrieval_and_ids_are_consistent(model, source):
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    table = next(t for t in model.model_list(kind="table") if t["data"]["name"] == "orders")
    assert model.get(table["id"]) == model.model_get(table["id"])
    model.begin("run", actor="test-user", question="Inspect orders", model="test-model", prompt_version="test")
    with pytest.raises(PermissionError):
        model.retrieve_table("run", "other-user", table["id"])
    assert not model.retrieved_tables("run", "test-user")
    assert model.retrieve_table("run", "test-user", table["id"])["columns"]
    catalog = model.query_catalog("run", "test-user", "sales")
    query = validate_sql(
        "SELECT SUM(amount) FROM orders",
        catalog,
        dialect="sqlite",
        retrieved=set(model.retrieved_tables("run", "test-user")),
    )
    assert source.run(query["sql"])["rows"] == [[150.0]]


def test_reviewed_term_survives_compilation_and_cas_guards(model, source):
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    term = model.model_list(kind="term")[0]
    edited = model.model_put(
        "term",
        dict(term["data"], description="Reviewed business meaning"),
        actor="editor",
        key=term["id"],
        expected=term["revision"],
    )
    reviewed = model.model_review(term["id"], actor="reviewer", expected=edited["revision"], certified=True)
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    assert model.model_get(term["id"]) == reviewed
    with pytest.raises(ValueError, match="revision conflict"):
        model.model_put(
            "term",
            dict(term["data"], description="Stale overwrite"),
            actor="editor",
            key=term["id"],
            expected=term["revision"],
        )
    with pytest.raises(ValueError, match="certification"):
        model.model_retire(term["id"], actor="editor", expected=reviewed["revision"])
    assert model.model_history(key=term["id"])[-1]["operation"] == "review"


def test_import_round_trip_atomicity_and_untrusted_certification(model, source, tmp_path):
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    term = model.model_list(kind="term")[0]
    model.model_review(term["id"], actor="reviewer", expected=term["revision"], certified=True)
    exported = model.model_export()
    other = open_ontology(tmp_path / "other", "test-user")
    assert other.model_import(exported, actor="importer")["imported"] == len(exported["objects"])
    assert other.model_get(term["id"])["certification"]["status"] == "draft"
    before = other.model_export()
    with pytest.raises(ValueError, match="unsupported"):
        other.model_import({"unknown": []}, actor="importer")
    assert other.model_export() == before


def test_semantic_vector_edit_invalidates_old_embedding(model):
    pytest.importorskip("sqlite_vec")
    first = model.model_put("term", {"name": "Turnover", "description": "Revenue"}, actor="editor")
    assert model.embedding_candidates([first["id"]], model="test-space") == [first]
    model.model_embed([{"id": first["id"], "revision": first["revision"], "vector": [1, 0, 0]}], model="test-space")
    assert model.embedding_candidates([first["id"]], model="test-space") == []
    assert model.model_search(vector=[1, 0, 0], model="test-space")[0]["object"]["id"] == first["id"]
    model.model_put(
        "term",
        {"name": "Turnover", "description": "Employee departures"},
        key=first["id"],
        expected=first["revision"],
        actor="editor",
    )
    assert model.model_search(vector=[1, 0, 0], model="test-space") == []
    assert model.embedding_candidates([first["id"]], model="test-space")
    with pytest.raises(ValueError):
        model.model_embed(
            [{"id": first["id"], "revision": model.model_get(first["id"])["revision"], "vector": [float("nan"), 0, 0]}],
            model="test-space",
        )
    assert model.embedding_candidates([first["id"]], model="test-space")


def test_nvidia_and_official_ossie_interchange(model, source, tmp_path):
    pytest.importorskip("ossie_nvidia_auto_ontology")
    from sharedagents_research.interchange import export_model, import_model

    model.compile_catalog("sales", source.catalog(), actor="test-user")
    native = export_model(model, "nvidia")
    assert len(native["semantic_layer"]["terms"]) == 2
    bundle = export_model(model, "ossie")
    assert bundle["ontology"] and bundle["mapping"] and bundle["semantic_model"]
    destination = open_ontology(tmp_path / "ossie-round-trip", "test-user")
    import_model(destination, bundle["semantic_model"], actor="importer")
    assert {obj["data"]["name"] for obj in destination.model_list(kind="term")} == {
        obj["data"]["name"] for obj in model.model_list(kind="term")
    }
    assert len(destination.model_list(kind="table")) == 2


def test_complete_native_semantic_draft_batches_and_pins_catalog(model, source):
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    assert model.compilation_status("sales")["status"] == "refinement_required"
    rows = []
    for table in model.model_list(kind="table"):
        context = model.table_context(table["id"])
        term = context["term"]
        rows.append(
            {
                "kind": "term",
                "id": term["id"],
                "expected_revision": term["revision"],
                "data": dict(
                    term["data"], description="Business records described by this source table and its declared keys."
                ),
            }
        )
        rows.extend(
            {
                "kind": "attribute",
                "data": {
                    "name": column["data"]["name"],
                    "description": "A source-backed field of the business record, with the catalog's declared type.",
                    "term": term["id"],
                    "column": column["id"],
                },
            }
            for column in context["columns"]
        )
    model.model_put_many(rows, actor="test-user")
    state = model.compilation_status("sales")
    assert state["status"] == "draft_ready" and state["ready_tables"] == 2
    assert state["semantic_correctness_certified"] is False
    attributes = model.model_list(kind="attribute")
    identifiers = [row for row in attributes if row["data"]["name"] == "id"]
    assert len(identifiers) == 2 and identifiers[0]["id"] != identifiers[1]["id"]
    model.begin(
        "metadata-run", actor="test-user", question="Document the source", model="test-model", prompt_version="test"
    )
    tables = model.model_list(kind="table")
    receipt = model.finish(
        "metadata-run",
        "test-user",
        answer="A provisional semantic draft over the observed catalog.",
        citations=[table["id"] for table in tables],
        limitations=["Business meaning still needs expert review."],
    )
    assert len(receipt["semantic_citations"]) == 2
    assert model.get(receipt["semantic_citations"][tables[0]["id"]])["data"] == tables[0]["data"]


def test_inferred_join_uses_source_validation_and_query_history(model, source):
    catalog = source.catalog()
    for table in catalog:
        table["foreign_keys"] = []
    analysis = source.analyze(catalog)
    relation = analysis["relationships"][0]
    assert relation["enabled"] and relation["validation"]["orphan_rows"] == 0
    assert relation["validation"]["coverage"] == "full_domain"
    model.compile_catalog(
        "sales",
        catalog,
        actor="test-user",
        analysis=analysis,
        query_history=[
            {"sql": "SELECT SUM(amount) FROM orders", "name": "Total orders"},
            {"sql": "DELETE FROM orders"},
        ],
    )
    assert len(model.model_list(kind="analysis")) == 1
    assert model.model_list(kind="analysis")[0]["data"]["executed"] is False
    assert model.model_list(kind="relationship")[0]["data"]["enabled"]
    with pytest.raises(ValueError, match="disabled"):
        obj = model.model_list(kind="relationship")[0]
        model.model_put("relationship", obj["data"], actor="agent", key=obj["id"], expected=obj["revision"])
    import sqlite3

    # The fixture connection is intentionally read-only; corrupt the test data
    # through its separate owner connection to check that inference rejects it.
    with sqlite3.connect(source.engine.url.database.removeprefix("file:")) as db:
        db.execute("INSERT INTO orders VALUES(3,999,4)")
    analysis = source.analyze(catalog)
    assert not analysis["relationships"][0]["enabled"]
    assert analysis["relationships"][0]["validation"]["orphan_rows"] == 1


def test_catalog_refresh_updates_schema_preserves_annotations_and_revokes_review(model, source):
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    table = model.model_list(kind="table")[0]
    edited = model.model_put(
        "table",
        dict(table["data"], description="Reviewed source context"),
        actor="editor",
        origin="human",
        key=table["id"],
        expected=table["revision"],
    )
    reviewed = model.model_review(table["id"], actor="reviewer", expected=edited["revision"], certified=True)
    catalog = source.catalog()
    for row in catalog:
        if row["name"] == table["data"]["name"]:
            row["columns"].append({"name": "new_observed_column", "type": "TEXT"})
    model.compile_catalog("sales", catalog, actor="test-user")
    changed = model.model_get(table["id"])
    assert changed["revision"] != reviewed["revision"]
    assert changed["certification"]["status"] == "draft"
    assert changed["data"]["description"] == "Reviewed source context"
    assert changed["data"]["columns"][-1]["name"] == "new_observed_column"
    with pytest.raises(ValueError, match="physical catalog"):
        model.model_put(
            "table",
            dict(changed["data"], columns=[]),
            actor="editor",
            origin="human",
            key=table["id"],
            expected=changed["revision"],
        )


def test_reviewed_rules_apply_to_compilation_but_preserve_reviewed_objects(model, source):
    tag = model.model_put("tag", {"name": "Financial"}, actor="reviewer")
    rule = model.model_put(
        "rule",
        {"name": "Order data", "target_kind": "table", "contains": "orders", "tags": [tag["id"]]},
        actor="reviewer",
    )
    model.model_review(rule["id"], actor="reviewer", expected=rule["revision"], certified=True)
    model.compile_catalog("sales", source.catalog(), actor="test-user")
    orders = next(t for t in model.model_list(kind="table") if t["data"]["name"] == "orders")
    assert orders["data"]["tags"] == [tag["id"]]


def test_executed_query_charts_and_csv_do_not_run_or_invent_data():
    from sharedagents_research.results import result_csv, result_view

    query = {
        "id": "query-id",
        "kind": "query",
        "result": {"columns": ["Customer", "Revenue"], "rows": [["=1+1", 150.0]], "truncated": False},
    }
    chart = result_view(query)
    assert chart["specification"]["data"]["values"] == [{"category": "=1+1", "value": 150.0}]
    assert "'=1+1" in result_csv(query)
    with pytest.raises(ValueError, match="numeric"):
        result_view(query, x="Revenue", y="Customer")
    with pytest.raises(ValueError, match="stored query"):
        result_csv({"kind": "term"})


@pytest.mark.asyncio
async def test_embedding_failure_is_explicit_and_does_not_claim_domain_rollback(model, monkeypatch):
    import json

    from sharedagents_research import native_tools

    model.begin("run", actor="test-user", question="Define sales", model="test-model", prompt_version="test")
    monkeypatch.setattr(
        native_tools, "context_store", lambda **kwargs: ({"run_id": "run", "actor": "test-user"}, model)
    )

    class Unavailable:
        provider, model, dims = "test", "test-model", 3

        async def embed_batch(self, texts):
            raise RuntimeError("Embedding service unavailable")

    monkeypatch.setattr(native_tools, "_embedder", lambda: Unavailable())
    result = await native_tools.run_tool(
        "ontology_update",
        {"action": "define", "kind": "term", "data": {"name": "Sales", "description": "Reported sales"}},
    )
    assert result.ok
    value = json.loads(result.content)
    assert value["indexing_complete"] is False and value["indexing_error"] == "RuntimeError"
    assert value["retrieval_mode"] == "lexical"
    assert model.model_get(value["id"])["data"]["name"] == "Sales"
    assert model.model_search("Sales")
    assert model.embedding_candidates([value["id"]], model=native_tools._embedding_space(Unavailable()))
