#!/usr/bin/env python3
"""Build the 33-family failure atlas from immutable retained V1 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.static_numerical_family_v2 import build_failure_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retained_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    retained = json.loads(args.retained_report.read_text())
    payload = {
        **build_failure_atlas(retained),
        "retained_report_path": str(args.retained_report),
        "retained_report_sha256": hashlib.sha256(args.retained_report.read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True)
    report = args.output / "report.json"
    report.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    (args.output / "sha256_manifest.json").write_text(json.dumps({
        "report.json": hashlib.sha256(report.read_bytes()).hexdigest(),
    }, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "families": len(payload["families"]),
        "v1_passed": payload["STATIC_FAMILY_V1"]["passed"],
        "v2": payload["STATIC_FAMILY_V2"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
