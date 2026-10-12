#!/usr/bin/env python3
"""Inspect the frozen prospective campaign; production execution is gated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "common_framework_release/PROSPECTIVE_CAMPAIGN.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--block", choices=tuple(f"P{i}" for i in range(1, 9)))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("production execution is disabled in this framework-release mission")
    if not MATRIX.is_file():
        raise SystemExit(f"campaign matrix absent; run build_common_framework_release.py: {MATRIX}")
    rows = [json.loads(line) for line in MATRIX.read_text().splitlines() if line.strip()]
    if args.block:
        rows = [row for row in rows if row["block"] == args.block]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["gate"]] = counts.get(row["gate"], 0) + 1
    print(json.dumps({"case_count": len(rows), "by_gate": counts,
                      "estimated_storage_GiB": sum(row["storage_estimate_GiB"] for row in rows),
                      "execution_authorized": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
