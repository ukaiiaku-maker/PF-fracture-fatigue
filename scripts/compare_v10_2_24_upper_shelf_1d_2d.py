#!/usr/bin/env python3
"""Compare v9.13 upper-shelf 1-D K50 predictions with v10.2.24 2-D results."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt


REFERENCE_RE = re.compile(r"K50_T(?P<T>[0-9]+)K_MPa_sqrt_m$")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty comparison: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-d-summary", type=Path, required=True)
    parser.add_argument("--one-d-reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    one_rows = _read_csv(args.one_d_reference)
    two_rows = _read_csv(args.two_d_summary)
    if not one_rows or not two_rows:
        raise RuntimeError("1-D reference and 2-D summary must both contain rows")

    reference_columns = []
    for name in one_rows[0]:
        match = REFERENCE_RE.match(name)
        if match:
            reference_columns.append((int(match.group("T")), name))
    reference_columns.sort()
    if not reference_columns:
        raise RuntimeError("1-D reference has no K50_T...K_MPa_sqrt_m columns")

    required_two = {"option_key", "candidate_id", "temperature_K", "K_50um_MPa_sqrt_m"}
    missing = sorted(required_two - set(two_rows[0]))
    if missing:
        raise RuntimeError(f"2-D summary missing columns: {missing}")

    two_lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in two_rows:
        key = (str(row["candidate_id"]), int(round(_number(row["temperature_K"]))))
        if key in two_lookup:
            raise RuntimeError(f"duplicate 2-D candidate/temperature row: {key}")
        two_lookup[key] = row

    merged: list[dict[str, object]] = []
    for row in one_rows:
        candidate = str(row["candidate_id"])
        for temperature, column in reference_columns:
            one_d = _number(row[column])
            two_row = two_lookup.get((candidate, temperature))
            two_d = (
                _number(two_row["K_50um_MPa_sqrt_m"])
                if two_row is not None
                else float("nan")
            )
            delta = two_d - one_d if math.isfinite(two_d) and math.isfinite(one_d) else float("nan")
            merged.append(
                {
                    "shelf_rank": int(round(_number(row["shelf_rank"]))),
                    "option_key": "" if two_row is None else two_row["option_key"],
                    "candidate_id": candidate,
                    "temperature_K": temperature,
                    "K50_1d_MPa_sqrt_m": one_d,
                    "K50_2d_MPa_sqrt_m": two_d,
                    "delta_K50_2d_minus_1d_MPa_sqrt_m": delta,
                    "absolute_error_MPa_sqrt_m": abs(delta) if math.isfinite(delta) else float("nan"),
                    "relative_error": (
                        delta / one_d
                        if math.isfinite(delta) and abs(one_d) > 0.0
                        else float("nan")
                    ),
                    "directional_dbtt_gain_1d_MPa_sqrt_m": _number(
                        row["y__directional_dbtt_gain"]
                    ),
                    "upper_shelf_1d_MPa_sqrt_m": _number(
                        row["y__high_temperature_plateau"]
                    ),
                    "peak_prominence_1d_MPa_sqrt_m": _number(
                        row["y__peak_prominence"]
                    ),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.out_dir / "v10_2_24_upper_shelf_1d_2d_K50_comparison.csv",
        merged,
    )

    by_candidate: dict[str, list[dict[str, object]]] = {}
    for row in merged:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)

    summaries: list[dict[str, object]] = []
    matched_cases = 0
    for candidate, local in by_candidate.items():
        local.sort(key=lambda row: int(row["temperature_K"]))
        finite = [
            row
            for row in local
            if math.isfinite(float(row["K50_1d_MPa_sqrt_m"]))
            and math.isfinite(float(row["K50_2d_MPa_sqrt_m"]))
        ]
        matched_cases += len(finite)
        errors = [float(row["delta_K50_2d_minus_1d_MPa_sqrt_m"]) for row in finite]
        absolute = [abs(value) for value in errors]
        summaries.append(
            {
                "shelf_rank": int(local[0]["shelf_rank"]),
                "candidate_id": candidate,
                "matched_temperatures": len(finite),
                "MAE_MPa_sqrt_m": sum(absolute) / len(absolute) if absolute else float("nan"),
                "RMSE_MPa_sqrt_m": (
                    math.sqrt(sum(value * value for value in errors) / len(errors))
                    if errors
                    else float("nan")
                ),
                "maximum_absolute_error_MPa_sqrt_m": max(absolute) if absolute else float("nan"),
            }
        )

        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        temperatures = [int(row["temperature_K"]) for row in local]
        ax.plot(
            temperatures,
            [float(row["K50_1d_MPa_sqrt_m"]) for row in local],
            marker="o",
            label="1-D",
        )
        ax.plot(
            temperatures,
            [float(row["K50_2d_MPa_sqrt_m"]) for row in local],
            marker="s",
            label="2-D",
        )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"$K_{50\,\mu m}$ (MPa$\sqrt{m}$)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{candidate}_K50_1d_vs_2d.png", dpi=240)
        fig.savefig(args.out_dir / f"{candidate}_K50_1d_vs_2d.pdf")
        plt.close(fig)

    summaries.sort(key=lambda row: int(row["shelf_rank"]))
    _write_csv(
        args.out_dir / "v10_2_24_upper_shelf_1d_2d_candidate_errors.csv",
        summaries,
    )
    print(
        "V10224_UPPER_SHELF_1D_2D_COMPARISON "
        f"candidates={len(summaries)} matched_cases={matched_cases} out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
