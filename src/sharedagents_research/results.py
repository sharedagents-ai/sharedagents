"""Portable charts and CSV from an actual, stored query result."""

import csv
import io
import math


def result_view(query, *, x=None, y=None, chart="bar"):
    if not query or query.get("kind") != "query":
        raise ValueError("an executed, stored query is required")
    if chart not in {"bar", "line", "scatter"}:
        raise ValueError("choose bar, line or scatter")
    result = query["result"]
    columns, rows = result["columns"], result["rows"]
    x = x if x is not None else (columns[0] if columns else None)
    y = (
        y
        if y is not None
        else next(
            (
                name
                for i, name in enumerate(columns)
                if rows
                and all(r[i] is None or isinstance(r[i], (float, int)) and not isinstance(r[i], bool) for r in rows)
                and name != x
            ),
            None,
        )
    )
    if x not in columns or y not in columns:
        raise ValueError("chart axes must name result columns with a numeric measure")
    xi, yi = columns.index(x), columns.index(y)
    if any(
        row[yi] is not None
        and (isinstance(row[yi], bool) or not isinstance(row[yi], (int, float)) or not math.isfinite(row[yi]))
        for row in rows
    ):
        raise ValueError("chart measure must contain finite numeric values or nulls")
    return {
        "query_id": query["id"],
        "truncated": result["truncated"],
        "specification": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "point" if chart == "scatter" else chart,
            "data": {"values": [{"category": row[xi], "value": row[yi]} for row in rows]},
            "encoding": {
                "x": {"field": "category", "type": "nominal", "title": x},
                "y": {"field": "value", "type": "quantitative", "title": y},
            },
        },
    }


def result_csv(query):
    if not query or query.get("kind") != "query":
        raise ValueError("an executed, stored query is required")
    output = io.StringIO(newline="")
    writer = csv.writer(output)

    def safe(value):
        # A downloaded CSV may be opened in a spreadsheet. Preserve values while
        # preventing source text from becoming an executable spreadsheet formula.
        return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value

    writer.writerow(map(safe, query["result"]["columns"]))
    writer.writerows([safe(value) for value in row] for row in query["result"]["rows"])
    return output.getvalue()
