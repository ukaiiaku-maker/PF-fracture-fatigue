#!/usr/bin/env python3
"""Build a reproducible same-material amplitude ladder from an acquisition row."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-factor", type=float,
                        help="Disambiguate a source containing several mapped factors.")
    parser.add_argument("--factors", required=True,
                        help="Comma-separated reference-load factors.")
    parser.add_argument("--exposure-target-cycles", type=float,
                        help="Attach/override a high-cycle exposure target.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.source.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["candidate_id"] == args.candidate_id and
                   (args.source_factor is None or
                    abs(float(row["reference_factor"]) - args.source_factor) < 1e-12)]
    if len(matches) != 1:
        raise SystemExit(f"expected one {args.candidate_id!r} row, found {len(matches)}")
    source = matches[0]
    parent_factor = float(source["reference_factor"])
    parent_kmax = float(source["Kmax_MPa_sqrt_m"])
    rows = []
    for rank, factor_text in enumerate(args.factors.split(","), 1):
        factor = float(factor_text)
        row = dict(source)
        row.update({
            "acquisition_rank": rank,
            "reference_factor": f"{factor:.12g}",
            "Kmax_MPa_sqrt_m": f"{parent_kmax * factor / parent_factor:.16g}",
            "ladder_parent_factor": f"{parent_factor:.12g}",
            "ladder_kind": "SAME_MATERIAL_TRANSITION_BRACKET",
        })
        if args.exposure_target_cycles is not None:
            row["exposure_target_cycles"] = f"{args.exposure_target_cycles:.12g}"
        rows.append(row)
    fields = list(rows[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows for {args.candidate_id} to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
