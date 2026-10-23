#!/usr/bin/env python3
"""Verify an extracted QSTE340TM active-calibration evidence package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/"PACKAGE_MANIFEST.json").read_text())
    failures = []
    for record in manifest["files"]:
        path = root/record["archive_path"]
        if not path.is_file():
            failures.append({"path": record["archive_path"], "reason": "missing"})
        elif path.stat().st_size != record["size_bytes"]:
            failures.append({"path": record["archive_path"], "reason": "size"})
        elif digest(path) != record["sha256"]:
            failures.append({"path": record["archive_path"], "reason": "sha256"})
    prediction = root/"results/tables/chronological_predictions.csv"
    with prediction.open(newline="", encoding="utf-8") as handle:
        prediction_rows = sum(1 for _ in csv.DictReader(handle))
    assessment = json.loads((root/"results/active_calibration_assessment.json").read_text())
    report = {
        "schema": manifest["schema"],
        "manifest_file_count": len(manifest["files"]),
        "failures": failures,
        "chronological_prediction_rows": prediction_rows,
        "source_csv_sha256": assessment["source_csv_sha256"],
        "passed": not failures and prediction_rows == 760,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
