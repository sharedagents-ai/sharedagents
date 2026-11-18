"""Read-only configured sources, schema profiling and catalog-grounded SQL.

Connection strings stay in environment/configuration, never in ontology objects
or model arguments. Driver imports are lazy; ontology policies live here rather
than in SharedData.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager

from .storage_client import configuration

# SQLAlchemy dialect/driver implementations supply the vendor protocols.
CONNECTORS = {
    "sqlite": "sqlite",
    "duckdb": "duckdb",
    "postgresql": "postgres",
    "mysql": "mysql",
    "snowflake": "snowflake",
    "databricks": "databricks",
    "clickhouse": "clickhouse",
    "clickhousedb": "clickhouse",
    "trino": "trino",
    "hive": "spark",
}


def configured_sources(scope):
    sources = configuration().get("connections", {})
    if not isinstance(sources, dict):
        raise ValueError("ontology connections configuration must be an object")
    return {
        name: value
        for name, value in sources.items()
        if isinstance(value, dict) and (scope == "operator" or scope in value.get("scopes", []))
    }


class Source:
    def __init__(self, name, *, scope, settings=None):
        from sqlalchemy import create_engine
        from sqlalchemy.engine import make_url

        self.name = name
        self.settings = settings if settings is not None else configured_sources(scope).get(name)
        if not isinstance(self.settings, dict):
            raise PermissionError("source is not configured for this ontology scope")
        variable = self.settings.get("url_env")
        if not isinstance(variable, str) or not variable or not os.environ.get(variable):
            raise ValueError("configured source requires its connection URL environment variable")
        try:
            url = make_url(os.environ[variable])
        except Exception:
            raise ValueError("configured source URL is malformed") from None
        aliases = {
            "postgres": "postgresql+psycopg",
            "postgresql": "postgresql+psycopg",
            "mysql": "mysql+mysqlconnector",
            "kyuubi": "hive",
        }
        if url.drivername in aliases:
            url = url.set(drivername=aliases[url.drivername])
        self.driver = url.drivername.split("+", 1)[0]
        if self.driver not in CONNECTORS:
            raise ValueError("unsupported connector dialect")
        self.dialect = CONNECTORS[self.driver]
        if self.driver not in {"sqlite", "duckdb"} and self.settings.get("read_only_credentials") is not True:
            raise ValueError("remote connectors require explicitly configured read-only database credentials")
        options = {"pool_pre_ping": True}
        if self.driver == "sqlite":
            if not url.database or url.database == ":memory:":
                raise ValueError("configure an existing SQLite database file")
            from pathlib import Path

            filename = Path(url.database).expanduser().resolve(strict=True)
            # URI quoting is delegated to SQLAlchemy's URL implementation.
            url = url.set(database="file:" + str(filename), query={"mode": "ro", "uri": "true"})
        if self.driver == "duckdb":
            options["connect_args"] = {"read_only": True, "config": {"enable_external_access": False}}
        try:
            self.engine = create_engine(url, **options)
        except Exception as exc:
            raise RuntimeError("configured connector initialization failed: " + type(exc).__name__) from None

    @contextmanager
    def connection(self):
        with self.engine.connect() as conn:
            if self.driver == "sqlite":
                conn.exec_driver_sql("PRAGMA query_only=ON")
                raw = conn.connection.driver_connection
                deadline = time.monotonic() + min(float(self.settings.get("timeout", 15)), 120)
                raw.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            elif self.driver == "postgresql":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(
                    "SET LOCAL statement_timeout = "
                    + str(max(100, min(int(self.settings.get("timeout", 15) * 1000), 120000)))
                )
            try:
                yield conn
            finally:
                if self.driver == "sqlite":
                    raw.set_progress_handler(None, 0)
                conn.rollback()

    def close(self):
        self.engine.dispose()

    def analyze(self, catalog):
        from .schema_analysis import analyze

        return analyze(self, catalog)

    def query_history(self):
        """Read an operator-selected history export, never an agent-selected path."""
        from pathlib import Path

        variable = self.settings.get("query_history_env")
        if not variable:
            return []
        path = Path(os.environ.get(variable, ""))
        if not path.is_file() or path.stat().st_size > 5_000_000:
            raise ValueError("configured query history must be an existing JSON file of at most 5 MB")
        value = json.loads(path.read_text())
        if (
            not isinstance(value, list)
            or len(value) > 1000
            or not all(
                isinstance(row, dict) and isinstance(row.get("sql"), str) and len(row["sql"]) <= 64000 for row in value
            )
        ):
            raise ValueError("query history must contain at most 1000 objects with bounded SQL strings")
        return value

    def catalog(self, *, samples=20):
        from sqlalchemy import inspect

        samples = max(0, min(int(samples), 100))
        result = []
        with self.connection() as conn:
            inspector = inspect(conn)
            schemas = self.settings.get("schemas") or [inspector.default_schema_name]
            for schema in schemas:
                names = [(name, "table") for name in inspector.get_table_names(schema=schema)]
                if self.settings.get("views", True):
                    names += [(name, "view") for name in inspector.get_view_names(schema=schema)]
                for name, kind in names:
                    allowed = self.settings.get("tables")
                    if allowed and name not in allowed:
                        continue
                    columns = inspector.get_columns(name, schema=schema)
                    preparer = conn.dialect.identifier_preparer
                    table_sql = (preparer.quote(schema) + "." if schema else "") + preparer.quote(name)
                    sample_rows = (
                        conn.exec_driver_sql("SELECT * FROM " + table_sql + " LIMIT " + str(samples)).fetchall()
                        if samples
                        else []
                    )
                    fields = []
                    for index, column in enumerate(columns):
                        values = [row[index] for row in sample_rows]
                        sensitive = bool(
                            re.search(r"email|password|secret|token|ssn|phone|address", column["name"], re.I)
                        )
                        # Sample-level PII hints are deterministic and explicitly
                        # incomplete; native research can refine them.
                        sensitive |= any(
                            isinstance(v, str) and re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", v) for v in values
                        )
                        visible = (
                            [] if sensitive else list(dict.fromkeys(str(v)[:200] for v in values if v is not None))[:20]
                        )
                        fields.append(
                            {
                                "name": column["name"],
                                "type": str(column["type"]),
                                "nullable": column.get("nullable", True),
                                "pii_suspected": sensitive,
                                "sample_values": visible,
                                "sample_count": len(values),
                                "sample_distinct": len(set(map(str, values))),
                                "sample_nulls": sum(v is None for v in values),
                            }
                        )
                    pk = (
                        inspector.get_pk_constraint(name, schema=schema).get("constrained_columns", [])
                        if kind == "table"
                        else []
                    )
                    foreign = inspector.get_foreign_keys(name, schema=schema) if kind == "table" else []
                    result.append(
                        {
                            "name": name,
                            "schema": schema or "",
                            "type": kind,
                            "columns": fields,
                            "primary_key": pk,
                            "foreign_keys": [
                                {
                                    "columns": f["constrained_columns"],
                                    "schema": f.get("referred_schema") or schema or "",
                                    "table": f["referred_table"],
                                    "target_columns": f["referred_columns"],
                                }
                                for f in foreign
                            ],
                        }
                    )
        return result

    def run(self, sql, *, limit=1000):
        limit = max(1, min(int(limit), 1000))
        started = time.monotonic()
        with self.connection() as conn:
            result = conn.exec_driver_sql(sql)
            rows = result.fetchmany(limit + 1)
            return {
                "columns": list(result.keys()),
                "rows": [[json.loads(json.dumps(value, default=str)) for value in row] for row in rows[:limit]],
                "truncated": len(rows) > limit,
                "elapsed_seconds": time.monotonic() - started,
            }


def validate_sql(sql, catalog, *, dialect, retrieved):
    """Accept one read query and resolve every table/column against the catalog."""
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError
    from sqlglot.optimizer.qualify import qualify

    if not isinstance(sql, str) or not sql.strip() or len(sql) > 64000:
        raise ValueError("SQL must contain 1..64000 characters")
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except SqlglotError as exc:
        raise ValueError("SQL parse failed: " + str(exc)) from None
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("only one read-only SELECT query is permitted")
    tree = statements[0]
    prohibited = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command, exp.Into, exp.Lock)
    if any(isinstance(node, prohibited) for node in tree.walk()):
        raise ValueError("SQL contains a non-read operation")
    safe_anonymous = {
        "date",
        "datetime",
        "strftime",
        "julianday",
        "ifnull",
        "iif",
        "date_trunc",
        "date_part",
        "dateadd",
        "datediff",
        "printf",
    }
    safe_functions = {
        "ABS",
        "AVG",
        "COUNT",
        "SUM",
        "MIN",
        "MAX",
        "CAST",
        "TRY_CAST",
        "COALESCE",
        "LOWER",
        "UPPER",
        "SUBSTRING",
        "CONCAT",
        "CONCAT_WS",
        "LENGTH",
        "CHAR_LENGTH",
        "ROUND",
        "TRIM",
        "LTRIM",
        "RTRIM",
        "IF",
        "CASE",
        "DATE",
        "DATETIME",
        "STRFTIME",
        "TIME_TO_STR",
        "TS_OR_DS_TO_TIMESTAMP",
        "TS_OR_DS_TO_DATE",
        "JULIANDAY",
        "DATE_TRUNC",
        "DATE_PART",
        "DATE_ADD",
        "DATE_SUB",
        "DATEDIFF",
        "DATE_DIFF",
        "EXTRACT",
        "YEAR",
        "MONTH",
        "DAY",
        "CURRENT_DATE",
        "CURRENT_TIMESTAMP",
        "ROW_NUMBER",
        "RANK",
        "DENSE_RANK",
        "LAG",
        "LEAD",
        "NTILE",
        "PERCENT_RANK",
        "STDDEV",
        "STDDEV_POP",
        "STDDEV_SAMP",
        "VARIANCE",
        "VAR_POP",
        "VAR_SAMP",
        "NULLIF",
        "GREATEST",
        "LEAST",
        "CEIL",
        "CEILING",
        "FLOOR",
        "POWER",
        "SQRT",
        "LOG",
        "LN",
        "EXP",
        "MOD",
        "REPLACE",
        "SPLIT_PART",
        "STRING_AGG",
        "GROUP_CONCAT",
        "ARRAY_AGG",
        "PERCENTILE_CONT",
        "PERCENTILE_DISC",
        "MEDIAN",
        "BOOL_AND",
        "BOOL_OR",
    }
    for function in tree.find_all(exp.Func):
        if not isinstance(function, exp.Anonymous) and function.sql_name() not in safe_functions:
            raise ValueError("SQL function is not approved for read-only analytics: " + function.sql_name())
    for node in tree.find_all(exp.Anonymous):
        if node.name.casefold() not in safe_anonymous:
            raise ValueError("SQL function is not approved for read-only analytics: " + node.name)
    ctes = {cte.alias_or_name.casefold() for cte in tree.find_all(exp.CTE)}
    schema, used = {}, []
    by_name = {}
    for table in catalog:
        data = table["data"]
        by_name[(data.get("schema", "").casefold(), data["name"].casefold())] = table
        schema.setdefault(data.get("schema", ""), {})[data["name"]] = {
            col["name"]: col["type"] for col in data["columns"]
        }
    for table in tree.find_all(exp.Table):
        if table.name.casefold() in ctes and not table.db:
            continue
        if table.catalog:
            raise ValueError("cross-connection SQL requires an explicitly configured federated source")
        matches = [
            obj
            for (namespace, name), obj in by_name.items()
            if name == table.name.casefold() and (not table.db or namespace == table.db.casefold())
        ]
        if len(matches) != 1:
            raise ValueError("SQL references an unknown or ambiguous catalog table: " + table.name)
        obj = matches[0]
        if obj["id"] not in retrieved:
            raise ValueError("retrieve the referenced catalog table before executing SQL")
        used.append(obj["id"])
    if not used:
        raise ValueError("an analytical query must reference a retrieved catalog table")
    # SQLGlot's resolver checks aliases, stars, subqueries and CTE columns.
    flat = {key: cols for namespace in schema.values() for key, cols in namespace.items()}
    resolution_schema = flat if len(schema) == 1 and next(iter(schema)) in {"", "main"} else schema
    try:
        qualified = qualify(
            tree.copy(), dialect=dialect, schema=resolution_schema, validate_qualify_columns=True, identify=False
        )
    except SqlglotError as exc:
        raise ValueError("SQL catalog validation failed: " + str(exc)) from None
    return {"sql": qualified.sql(dialect=dialect), "tables": sorted(set(used)), "validated": True}
