#!/usr/bin/env python3
"""Inventory external artifact-bound donor tests without copying fixtures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for path in sorted(args.source.glob("test_*.py")):
        data = path.read_bytes()
        rows.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                     "bytes": len(data)})
    payload = {
        "schema": "omitted-artifact-tests/1", "status": "external_not_collected",
        "source_directory": str(args.source), "count": len(rows), "tests": rows,
        "restore_instruction": (
            "Restore the recorded donor test source plus its named external fixtures, "
            "verify these hashes, then collect explicitly; the fast suite does not download artifacts."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if len(rows) == 88 else 2

if __name__ == "__main__":
    raise SystemExit(main())
