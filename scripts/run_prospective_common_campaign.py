#!/usr/bin/env python3
"""Inspect the current frozen v3 campaign; production execution is gated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "common_framework_readiness/PROSPECTIVE_CAMPAIGN_V3.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--block", choices=tuple(f"P{i}" for i in range(1, 10)))
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("production execution is disabled in this framework-release mission")
    if not MATRIX.is_file():
        raise SystemExit(
            "current campaign matrix absent; run "
            f"scripts/build_prospective_campaign_v3.py --disabled --audit --dry-run: {MATRIX}"
        )
    rows = [json.loads(line) for line in MATRIX.read_text().splitlines() if line.strip()]
    if args.resume_state is not None:
        state = json.loads(args.resume_state.read_text())
        completed = set(state.get("completed_case_ids", []))
        rows = [row for row in rows if row["case_id"] not in completed]
    if args.block:
        rows = [row for row in rows if row["block"] == args.block]
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for axis, value in row["readiness"].items():
            axis_counts = counts.setdefault(axis, {})
            axis_counts[value] = axis_counts.get(value, 0) + 1
    print(json.dumps({
        "schema": "prospective-common-fracture-campaign-status/3",
        "campaign": str(MATRIX),
        "case_count": len(rows),
        "selected_block": args.block,
        "resume_state": None if args.resume_state is None else str(args.resume_state),
        "by_readiness_axis": counts,
        "estimated_storage_GiB": sum(
            float(row.get(
                "storage_estimate_GiB",
                row.get("resource_estimate", {}).get("retained_output_GiB", 0.0),
            ))
            for row in rows
        ),
        "native_command_count": sum(bool(row.get("native_args")) for row in rows),
        "execution_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
