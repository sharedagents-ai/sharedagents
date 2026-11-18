"""One private data-layer operation; invoked only by the native tool adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .ontology import encoded, open_ontology


def main():
    try:
        raw = sys.stdin.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("ontology request exceeds the operation size limit")
        request = json.loads(raw)
        action = request["action"]
        if action not in {
            "begin",
            "check_open",
            "capture",
            "extract",
            "finish",
            "search",
            "get",
            "describe",
            "status",
            "timeline",
            "export",
        }:
            raise ValueError("unknown ontology operation")
        store = open_ontology(Path(request["root"]), request["scope"], create=bool(request["create"]))
        value = getattr(store, action)(*request.get("args", []), **request.get("kwargs", {}))
        response = {"ok": True, "value": value}
    except Exception as exc:
        response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    print(encoded(response))


if __name__ == "__main__":
    main()
