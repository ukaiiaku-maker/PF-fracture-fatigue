#!/usr/bin/env python3
"""Build/audit the disabled v3 campaign; never launch production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.prospective_campaign_v3 import build_cases_v3, summary


OUT = ROOT / "common_framework_readiness"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disabled", action="store_true", required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--block", action="append", default=[])
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    rows = build_cases_v3(ROOT)
    if args.block:
        requested = set(args.block)
        rows = [row for row in rows if row["block"] in requested]
        absent = requested - {row["block"] for row in rows}
        if absent:
            raise SystemExit(f"unknown/empty selected blocks: {sorted(absent)}")
    if args.resume_state is not None:
        state = json.loads(args.resume_state.read_text())
        completed = set(state.get("completed_case_ids", []))
        rows = [row for row in rows if row["case_id"] not in completed]

    raw = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    report = summary(rows, raw)
    if not report["validation"]["valid"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "PROSPECTIVE_CAMPAIGN_V3.jsonl").write_bytes(raw)
    (args.output_dir / "CAMPAIGN_V3_SUMMARY.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
