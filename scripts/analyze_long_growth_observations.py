#!/usr/bin/env python3
"""Re-index protected long-growth steps files with the stage-aware observer."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from arrhenius_fracture.fracture_observation_history import from_legacy_steps_csv


def analyze(root: Path, targets_um: tuple[float, ...]) -> list[dict]:
    records = []
    for path in sorted(root.rglob("steps_*K.csv")):
        history = from_legacy_steps_csv(path)
        if not history.rows:
            continue
        achieved = max(row.committed_length_m for row in history.rows)
        record = {
            "case": str(path.parent.relative_to(root)),
            "steps_file": str(path),
            "event_count": len(history.rows),
            "achieved_extension_um": achieved * 1e6,
            "queries": {},
        }
        for target_um in targets_um:
            result = history.at_length(target_um * 1e-6)
            candidates = result.exact or result.crossing
            record["queries"][f"{target_um:g}_um"] = {
                "status": result.status,
                "K_I_MPa_sqrt_m": [row.K_I_Pa_sqrt_m * 1e-6 for row in candidates],
                "event_ids": [row.event_id for row in candidates],
                "interpolation_performed": result.interpolation_performed,
            }
        records.append(record)
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--targets-um", type=float, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = {
        "schema": "long-growth-observation-reanalysis/1",
        "source_root": str(args.root.resolve()),
        "targets_um": args.targets_um,
        "cases": analyze(args.root, tuple(args.targets_um)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
