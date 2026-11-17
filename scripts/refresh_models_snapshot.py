#!/usr/bin/env python3
"""Regenerate the bundled models snapshot from live Artificial Analysis data.

Pulls the current AA Intelligence Index + cost-per-task table — the same
endpoint the Models page uses — and rewrites
``src/sharedagents_web/data/models_snapshot.json``, the static fallback served to a
fresh sharedagents install that has no ``ARTIFICIAL_ANALYSIS_API_KEY``.

The snapshot is the raw AA payload in cache format (base rows + metadata) plus
two provenance fields: ``snapshot_date`` (the fetch date, ``YYYY-MM-DD``) and
``source: "snapshot"``. It is serialised with ``sort_keys=True, indent=2`` so a
re-run over unchanged data is a byte-for-byte no-op (idempotent).

Requires ``ARTIFICIAL_ANALYSIS_API_KEY`` in the environment (or
``~/.sharedagents/.env``). Dry-run by default; ``--apply`` writes.

Examples
--------
    python scripts/refresh_models_snapshot.py            # dry run: show the diff
    python scripts/refresh_models_snapshot.py --apply    # write the snapshot
    python scripts/refresh_models_snapshot.py --out /tmp/snap.json --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from sharedagents_web.model_benchmarks import (  # noqa: E402
    BenchmarkError,
    _fetch_all_pages,
    _normalize_raw,
    api_key,
)

_SNAPSHOT_REL = Path("src") / "sharedagents_web" / "data" / "models_snapshot.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_snapshot() -> dict[str, Any]:
    """Fetch live AA data and shape it into the bundled-snapshot payload.

    Reuses the exact building blocks the live path uses
    (``_fetch_all_pages`` -> ``_normalize_raw``), so the result is byte-for-byte
    the same base format the page's cache stores — with provenance added.
    """
    key = api_key()
    if not key:
        raise SystemExit(
            "no ARTIFICIAL_ANALYSIS_API_KEY in the environment or ~/.sharedagents/.env — set it to pull live data."
        )
    raw = _fetch_all_pages(key, None)
    base = _normalize_raw(raw)
    base["snapshot_date"] = str(base.get("updated_at") or "")[:10]
    base["source"] = "snapshot"
    return base


def _serialize(payload: dict[str, Any]) -> str:
    # sort_keys keeps the output deterministic so a no-op re-run is byte-identical.
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _read_existing(path: Path) -> tuple[str | None, int | None, str | None]:
    """Return ``(text, model_count, snapshot_date)`` for an existing snapshot."""
    if not path.exists():
        return None, None, None
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except ValueError:
        return text, None, None  # unparseable existing file -> treat as a rewrite
    return (
        text,
        len(payload.get("models") or []),
        str(payload.get("snapshot_date") or ""),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--apply", action="store_true", help="write the snapshot (default: dry run)")
    ap.add_argument("--out", default="", help="snapshot path (default: the bundled data file)")
    args = ap.parse_args(argv)

    snapshot_path = Path(args.out).expanduser() if args.out else _repo_root() / _SNAPSHOT_REL

    try:
        payload = _build_snapshot()
    except BenchmarkError as exc:
        print(f"live fetch failed ({exc.code}): {exc.message}")
        return 1

    new_text = _serialize(payload)
    new_models = len(payload.get("models") or [])
    new_date = payload.get("snapshot_date")

    old_text, old_models, old_date = _read_existing(snapshot_path)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] target={snapshot_path}")
    print(f"  live fetch : models={new_models}  snapshot_date={new_date}")
    if old_models is not None:
        print(f"  existing   : models={old_models}  snapshot_date={old_date}")
    else:
        print("  existing   : (none)")

    if old_text == new_text:
        print("  no change — snapshot already up to date (idempotent no-op)")
        return 0

    if args.apply:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(new_text, encoding="utf-8")
        print(f"  wrote {len(new_text)} bytes ({new_models} models)")
    else:
        old_bytes = len(old_text) if old_text is not None else 0
        print(f"  would change: {old_bytes} -> {len(new_text)} bytes")
        print("  (dry run — pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
