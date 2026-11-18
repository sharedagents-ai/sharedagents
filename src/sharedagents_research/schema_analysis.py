"""Conservative schema families, junctions and source-validated inferred joins."""

import re
from collections import defaultdict


def analyze(source, catalog, *, max_candidates=50):
    """Name similarity proposes joins; actual source queries decide enablement.

    A successful validation checks the entire key domain under the connector's
    read-only deadline. A timeout or unsupported query leaves a disabled draft.
    No inferred relationship is certified by this operation.
    """
    families = defaultdict(list)
    by_name = {(t.get("schema", ""), t["name"]): t for t in catalog}
    junctions, relationships = [], []
    for table in catalog:
        signature = tuple(sorted((c["name"].casefold(), c["type"].casefold()) for c in table["columns"]))
        families[signature].append([table.get("schema", ""), table["name"]])
        foreign = table.get("foreign_keys", [])
        keys = set(table.get("primary_key", []))
        if len(foreign) >= 2 and keys and keys <= {c for fk in foreign for c in fk["columns"]}:
            junctions.append({"table": [table.get("schema", ""), table["name"]], "foreign_keys": foreign})
    with source.connection() as conn:
        quote = conn.dialect.identifier_preparer.quote

        def sql_table(table):
            return (quote(table["schema"]) + "." if table.get("schema") else "") + quote(table["name"])

        for table in catalog:
            declared = {tuple(fk["columns"]) for fk in table.get("foreign_keys", [])}
            for column in table["columns"]:
                if (column["name"],) in declared:
                    continue
                name = re.sub(r"[^a-z0-9]", "", column["name"].casefold())
                for target in by_name.values():
                    pk = target.get("primary_key", [])
                    if target is table or len(pk) != 1 or len(relationships) >= max_candidates:
                        continue
                    target_name = re.sub(r"[^a-z0-9]", "", target["name"].casefold()).removesuffix("s")
                    if name not in {target_name + "id", target_name + re.sub(r"[^a-z0-9]", "", pk[0].casefold())}:
                        continue
                    data = {
                        "source_table": [table.get("schema", ""), table["name"]],
                        "target_table": [target.get("schema", ""), target["name"]],
                        "columns": [column["name"]],
                        "target_columns": pk,
                        "declared": False,
                        "enabled": False,
                        "validation": {"status": "unverified", "coverage": "full_domain"},
                    }
                    a, b = sql_table(table), sql_table(target)
                    left, right = quote(column["name"]), quote(pk[0])
                    try:
                        duplicates = conn.exec_driver_sql(
                            f"SELECT COUNT(*) FROM (SELECT {right} FROM {b} WHERE {right} IS NOT NULL "
                            f"GROUP BY {right} HAVING COUNT(*)>1) duplicate_keys"
                        ).scalar_one()
                        nonnull = conn.exec_driver_sql(
                            f"SELECT COUNT(*) FROM {a} WHERE {left} IS NOT NULL"
                        ).scalar_one()
                        orphans = conn.exec_driver_sql(
                            f"SELECT COUNT(*) FROM {a} a WHERE a.{left} IS NOT NULL AND NOT EXISTS "
                            f"(SELECT 1 FROM {b} b WHERE b.{right}=a.{left})"
                        ).scalar_one()
                        valid = not duplicates and not orphans and bool(nonnull)
                        data["enabled"] = valid
                        data["validation"].update(
                            status="validated" if valid else "rejected",
                            source_nonnull=nonnull,
                            orphan_rows=orphans,
                            duplicate_target_keys=duplicates,
                        )
                    except Exception as exc:
                        # Do not put connection strings or query values in metadata.
                        data["validation"].update(status="unverified", error_type=type(exc).__name__)
                    relationships.append(data)
    return {
        "families": [members for members in families.values() if len(members) > 1],
        "junctions": junctions,
        "relationships": relationships,
        "candidate_limit": max_candidates,
    }
