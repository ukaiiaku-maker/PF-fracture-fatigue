#!/usr/bin/env python3
"""Build no-refit protocol holdouts at one fixed absolute peak load.

The HCF runner consumes a temperature-specific reference factor.  This utility
changes that encoding factor only so that every row retains exactly the same
absolute Kmax; it does not alter any material parameter or fit a protocol.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_r10_coupled_opening_search import reference_load


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--Kmax", type=float, required=True,
                        help="Fixed absolute peak load in MPa sqrt(m).")
    parser.add_argument("--horizon", type=float, default=1.0e6)
    parser.add_argument("--phases", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.registry.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one {args.candidate_id!r} row, found {len(matches)}")
    material = matches[0]

    # The 300 K, R=0.1, 1000 Hz baseline is intentionally not repeated.
    protocols = [
        (600.0, 0.1, 1000.0, "temperature_holdout"),
        (900.0, 0.1, 1000.0, "temperature_holdout"),
        (1200.0, 0.1, 1000.0, "temperature_holdout"),
        (300.0, 0.3, 1000.0, "load_ratio_holdout"),
        (300.0, 0.5, 1000.0, "load_ratio_holdout"),
        (300.0, 0.1, 10.0, "frequency_holdout"),
        (300.0, 0.1, 100.0, "frequency_holdout"),
    ]
    rows: list[dict[str, object]] = []
    for temperature, load_ratio, frequency, role in protocols:
        reference, _ = reference_load(material, temperature, args.phases)
        rows.append({
            **material,
            "temperature_K": temperature,
            "R": load_ratio,
            "frequency_Hz": frequency,
            "reference_factor": args.Kmax / reference,
            "target_absolute_Kmax_MPa_sqrt_m": args.Kmax,
            "exposure_target_cycles": args.horizon,
            "selection_basis": "FIXED_ABSOLUTE_KMAX_NO_REFIT",
            "protocol_role": role,
        })
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
