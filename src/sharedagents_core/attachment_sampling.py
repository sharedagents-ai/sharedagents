"""Bounded CSV/TSV inspection without loading a dataset into model context.

Run ``python -m sharedagents_core.attachment_sampling <path> ...``. Initial records
are preserved in order: a file may contain metadata before its column headers.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

MAX_READ_BYTES = 64 * 1024
MAX_OUTPUT_CHARS = 12_000


def sample_csv(path: Path) -> dict:
    """Read at most 64 KiB and return up to five initial records."""
    with path.open("rb") as handle:
        raw = handle.read(MAX_READ_BYTES)
    text = raw.decode("utf-8-sig", errors="replace")
    partial = path.stat().st_size > len(raw)
    if partial:
        text = text.rsplit("\n", 1)[0] if "\n" in text else ""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    records = []
    reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
    error = None
    for _ in range(5):
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            error = str(exc)
            break
        records.append(
            {
                "columns": len(row),
                "values": [cell[:160] for cell in row[:24]],
                "clipped": len(row) > 24 or any(len(cell) > 160 for cell in row[:24]),
            }
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "bytes_read": len(raw),
        "initial_records": records,
        "sample_only": True,
        "input_truncated": partial,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.paths:
        try:
            result = sample_csv(path)
        except (OSError, ValueError) as exc:
            result = {"path": str(path), "error": str(exc)}
        candidate = results + [result]
        output = {"files": candidate, "remaining_files": len(args.paths) - len(candidate)}
        if len(json.dumps(output, ensure_ascii=False)) > MAX_OUTPUT_CHARS:
            break
        results = candidate
    print(
        json.dumps(
            {
                "files": results,
                "remaining_files": len(args.paths) - len(results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
