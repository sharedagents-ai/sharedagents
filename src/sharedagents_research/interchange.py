"""Native/NVIDIA model interchange and Apache Ossie conversion.

Apache's official converter owns Ossie syntax. This adapter maps SharedAgents'
model records onto its documented NVIDIA input rather than inventing an Ossie
lookalike. Evidence and publication history remain separate from model export.
"""

from __future__ import annotations

from .ontology import identity


def nvidia_document(store):
    objects = store.model_export()["objects"]
    by_id = {obj["id"]: obj for obj in objects}
    databases, table_columns, foreign, joins = {}, {}, [], []
    for obj in objects:
        if obj["kind"] != "table":
            continue
        data, key = obj["data"], obj["id"]
        connection, schema = data["connection"], data.get("schema", "")
        database = databases.setdefault(
            connection, {"id": connection, "dialect": data.get("dialect", ""), "schemas": {}}
        )
        namespace = database["schemas"].setdefault(
            schema,
            {"id": identity("schema", connection, schema), "name": schema, "database_name": connection, "tables": []},
        )
        columns = [
            {
                "id": identity("column", key, column["name"]),
                "name": column["name"],
                "description": column.get("description", ""),
                "type": column.get("type", ""),
                "sample_values": column.get("sample_values", []),
                "is_nullable": column.get("nullable", True),
                "is_unique": column["name"] in data.get("primary_key", []),
            }
            for column in data["columns"]
        ]
        table_columns[key] = columns
        namespace["tables"].append(
            {
                "id": key,
                "name": data["name"],
                "description": data.get("description", ""),
                "pk": data.get("primary_key", []),
                "type": data.get("type", "table"),
                "columns": columns,
            }
        )
    terms, metrics, analyses, semantic_fks = [], [], [], []
    for obj in objects:
        data = obj["data"]
        if obj["kind"] == "term":
            attributes = []
            mapped = data.get("tables", [])
            for table in mapped:
                for column in table_columns.get(table, []):
                    attributes.append(
                        {
                            "id": identity("attribute", obj["id"], column["id"]),
                            "name": column["name"],
                            "description": column.get("description", ""),
                            "column_id": column["id"],
                        }
                    )
            explicit = [row for row in objects if row["kind"] == "attribute" and row["data"].get("term") == obj["id"]]
            for row in explicit:
                attributes = [
                    attribute for attribute in attributes if attribute["column_id"] != row["data"].get("column")
                ]
                attributes.append(
                    {
                        "id": row["id"],
                        "name": row["data"]["name"],
                        "description": row["data"].get("description", ""),
                        "column_id": row["data"]["column"],
                    }
                )
            terms.append(
                {
                    "id": obj["id"],
                    "name": data["name"],
                    "description": data.get("description", ""),
                    "represents": mapped,
                    "columns_attributes": attributes,
                }
            )
        if obj["kind"] in {"metric", "analysis"}:
            row = {
                "id": obj["id"],
                "name": data["name"],
                "description": data.get("description", ""),
                "sql": data.get("sql", ""),
                "sql_column_is": data.get("columns", []),
            }
            if obj["kind"] == "metric":
                row["term_id"] = data.get("term", "")
                metrics.append(row)
            else:
                analyses.append(row)
        if obj["kind"] == "relationship" and data.get("columns") and data.get("target_columns"):
            source, target = data["source"], data["target"]
            if source not in by_id or target not in by_id:
                raise ValueError("relationship has an unavailable endpoint")
            pairs = list(zip(data["columns"], data["target_columns"]))
            joins.append(
                {
                    "source_table_id": source,
                    "target_table_id": target,
                    "join_columns": [{"source": a, "target": b} for a, b in pairs],
                }
            )
            foreign.extend(
                {"source_column_id": identity("column", source, a), "target_column_id": identity("column", target, b)}
                for a, b in pairs
            )
    return {
        "data_layer": {
            "databases": [
                dict(database, schemas=list(database["schemas"].values())) for database in databases.values()
            ],
            "foreign_keys": foreign,
            "joins": joins,
        },
        "semantic_layer": {
            "terms": terms,
            "semantic_fks": semantic_fks,
            "sql_attributes": {"manual": metrics, "table": [], "sql": [], "bridge_table": []},
            "custom_analyses": analyses,
        },
        "zones": [{"id": obj["id"], **obj["data"]} for obj in objects if obj["kind"] == "zone"],
    }


def export_model(store, format="native"):
    if format == "native":
        return store.model_export()
    document = nvidia_document(store)
    if format == "nvidia":
        return document
    if format != "ossie":
        raise ValueError("unknown semantic model export format")
    import yaml

    try:
        from ossie_nvidia_auto_ontology import convert_auto_ontology_to_ossie, convert_auto_ontology_to_ossie_ontology
    except ImportError as exc:
        raise RuntimeError("install Apache's official NVIDIA Ossie converter to use Ossie interchange") from exc
    text = yaml.safe_dump(document, sort_keys=False)
    ontology, mapping = convert_auto_ontology_to_ossie_ontology(text)
    return {
        "format": "ossie",
        "semantic_model": convert_auto_ontology_to_ossie(text),
        "ontology": ontology,
        "mapping": mapping,
    }


def import_model(store, document, *, actor):
    if isinstance(document, str):
        import yaml

        if len(document) > 8 * 1024 * 1024:
            raise ValueError("model document exceeds the size limit")
        for index, event in enumerate(yaml.parse(document)):
            if isinstance(event, yaml.AliasEvent) or index > 200000:
                raise ValueError("YAML aliases and oversized model documents are not accepted")
        document = yaml.safe_load(document)
    if not isinstance(document, dict):
        raise ValueError("semantic model document must be an object")
    if document.get("format") == "sharedagents-semantic-v1":
        return store.model_import(document, actor=actor)
    if "data_layer" not in document or "semantic_layer" not in document:
        if "datasets" not in document:
            raise ValueError("unsupported semantic model document")
        import yaml

        try:
            from ossie_nvidia_auto_ontology import convert_ossie_to_auto_ontology
        except ImportError as exc:
            raise RuntimeError("install Apache's official NVIDIA Ossie converter to import Ossie") from exc
        document = yaml.safe_load(convert_ossie_to_auto_ontology(yaml.safe_dump(document)))
    objects, column_ids, table_ids = [], {}, {}
    for database in document["data_layer"]["databases"]:
        connection = database["id"]
        for schema in database.get("schemas", []):
            for table in schema.get("tables", []):
                key = identity("catalog", connection, schema["name"], table["name"])
                table_ids[table["id"]] = key
                columns = [
                    {
                        "name": col["name"],
                        "type": col.get("type", ""),
                        "nullable": col.get("is_nullable", True),
                        "sample_values": col.get("sample_values", []),
                    }
                    for col in table.get("columns", [])
                ]
                objects.append(
                    {
                        "id": key,
                        "kind": "table",
                        "data": {
                            "name": table["name"],
                            "schema": schema["name"],
                            "connection": connection,
                            "dialect": database.get("dialect", ""),
                            "columns": columns,
                            "primary_key": table.get("pk", []),
                            "foreign_keys": [],
                        },
                    }
                )
                for col in table.get("columns", []):
                    column_key = identity("column", key, col["name"])
                    column_ids[col["id"]] = column_key
                    objects.append(
                        {
                            "id": column_key,
                            "kind": "column",
                            "data": {
                                "name": col["name"],
                                "description": col.get("description", ""),
                                "type": col.get("type", ""),
                                "table": key,
                                "connection": connection,
                            },
                        }
                    )
    semantic = document["semantic_layer"]
    for term in semantic.get("terms", []):
        objects.append(
            {
                "id": term["id"],
                "kind": "term",
                "data": {
                    "name": term["name"],
                    "description": term.get("description", ""),
                    "tables": [table_ids[key] for key in term.get("represents", [])],
                },
            }
        )
        for attribute in term.get("columns_attributes", []):
            objects.append(
                {
                    "id": attribute["id"],
                    "kind": "attribute",
                    "data": {
                        "name": attribute["name"],
                        "description": attribute.get("description", ""),
                        "term": term["id"],
                        "column": column_ids[attribute["column_id"]],
                    },
                }
            )
    for kind, rows in (
        ("metric", [row for group in semantic.get("sql_attributes", {}).values() for row in group]),
        ("analysis", semantic.get("custom_analyses", [])),
    ):
        for row in rows:
            data = {"name": row["name"], "description": row.get("description", ""), "sql": row.get("sql", "")}
            if row.get("term_id"):
                data["term"] = row["term_id"]
            objects.append({"id": row["id"], "kind": kind, "data": data})
    for join in document["data_layer"].get("joins", []):
        source, target = table_ids[join["source_table_id"]], table_ids[join["target_table_id"]]
        objects.append(
            {
                "id": identity("imported-join", source, target, join["join_columns"]),
                "kind": "relationship",
                "data": {
                    "name": "Imported relationship",
                    "source": source,
                    "target": target,
                    "columns": [pair["source"] for pair in join["join_columns"]],
                    "target_columns": [pair["target"] for pair in join["join_columns"]],
                    "enabled": False,
                    "validation_required": True,
                },
            }
        )
    for zone in document.get("zones", []):
        objects.append(
            {"id": zone["id"], "kind": "zone", "data": {key: value for key, value in zone.items() if key != "id"}}
        )
    return store.model_import({"format": "sharedagents-semantic-v1", "objects": objects}, actor=actor)
