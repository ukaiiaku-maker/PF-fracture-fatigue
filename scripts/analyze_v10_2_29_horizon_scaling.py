#!/usr/bin/env python3
"""Audit whether fixed-DeltaK VHCF runs scale with state changes, not cycle horizon."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCHEMA = "v10.2.29_vhcf_horizon_scaling_v1"


def read_rows(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--max-records-per-case", type=int, default=1000)
    parser.add_argument("--require-censored-horizon", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    tag = f"{int(round(args.temperature_K)):04d}K"

    cases: list[dict] = []
    for control_path in sorted(root.rglob("v10_2_29_fixed_deltaK_control.json")):
        run_root = control_path.parent
        steps_path = run_root / f"steps_{tag}.csv"
        if not steps_path.is_file():
            continue
        control = json.loads(control_path.read_text())
        rows = read_rows(steps_path)
        names = set(rows.dtype.names or ())
        if "fatigue_cycles" not in names:
            raise ValueError(f"{steps_path} lacks fatigue_cycles")
        blocks = np.maximum(np.asarray(rows["fatigue_cycles"], float), 0.0)
        cycles_total = float(np.sum(blocks))
        event_count = (
            int(np.sum(np.asarray(rows["n_fire"], float) > 0.0))
            if "n_fire" in names else 0
        )
        horizon = float(control.get("cycles_max", 0.0))
        censored = event_count == 0
        horizon_reached = bool(
            horizon > 0.0 and cycles_total >= horizon * (1.0 - 1.0e-10)
        )
        positive = blocks[blocks > 0.0]
        cases.append({
            "run_root": str(run_root),
            "parameter_option": control.get("parameter_option"),
            "target_deltaK_MPa_sqrt_m": float(
                control["target_deltaK_MPa_sqrt_m"]
            ),
            "cycles_horizon": horizon,
            "cycles_consumed": cycles_total,
            "horizon_reached": horizon_reached,
            "event_count": event_count,
            "right_censored": censored,
            "accepted_block_count": int(len(rows)),
            "positive_block_count": int(positive.size),
            "maximum_cycles_per_block": (
                float(np.max(positive)) if positive.size else 0.0
            ),
            "median_cycles_per_block": (
                float(np.median(positive)) if positive.size else 0.0
            ),
            "mean_cycles_per_block": (
                float(np.mean(positive)) if positive.size else 0.0
            ),
            "records_within_gate": int(len(rows)) <= args.max_records_per_case,
        })

    if not cases:
        raise SystemExit(f"no VHCF horizon cases found below {root}")

    failures = []
    for case in cases:
        if not case["records_within_gate"]:
            failures.append(
                f"{case['run_root']}: {case['accepted_block_count']} records"
            )
        if args.require_censored_horizon:
            if not case["right_censored"]:
                failures.append(f"{case['run_root']}: event occurred")
            elif not case["horizon_reached"]:
                failures.append(
                    f"{case['run_root']}: horizon not reached "
                    f"({case['cycles_consumed']} < {case['cycles_horizon']})"
                )

    csv_path = root / "vhcf_horizon_scaling.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    payload = {
        "schema": SCHEMA,
        "root": str(root),
        "temperature_K": float(args.temperature_K),
        "max_records_per_case": int(args.max_records_per_case),
        "require_censored_horizon": bool(args.require_censored_horizon),
        "case_count": len(cases),
        "pass": not failures,
        "failures": failures,
        "cases": cases,
    }
    (root / "vhcf_horizon_scaling.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("ERROR: VHCF horizon-scaling gate failed")


if __name__ == "__main__":
    main()
