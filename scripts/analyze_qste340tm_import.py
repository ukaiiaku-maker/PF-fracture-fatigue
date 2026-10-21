#!/usr/bin/env python3
"""Characterize the imported QSTE340TM observations without fitting a model.

This script deliberately stays on the observation side of the model boundary. It
does not infer temperature, frequency, local stress, crack closure, or a G* barrier.
It produces deterministic curve summaries and protocol-level holdout definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    if len(xs) < 3:
        raise ValueError("at least three points are required")
    xbar, ybar = fmean(xs), fmean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if sxx == 0.0:
        raise ValueError("degenerate x interval")
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(value * value for value in residuals)
    syy = sum((y - ybar) ** 2 for y in ys)
    slope_se = math.sqrt((sse / (len(xs) - 2)) / sxx)
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_standard_error": slope_se,
        "r_squared": 1.0 - sse / syy if syy else 1.0,
        "rmse_log10_rate": math.sqrt(sse / len(xs)),
    }


def summarize_segment(rows: list[dict[str, str]]) -> dict[str, float | int]:
    delta_k = [float(row["reported_DeltaK_MPa_sqrt_m"]) for row in rows]
    rates = [float(row["dadn_m_per_cycle"]) for row in rows]
    fit = linear_fit([math.log10(x) for x in delta_k], [math.log10(y) for y in rates])
    return {
        "points": len(rows),
        "source_order_first": int(rows[0]["source_order"]),
        "source_order_last": int(rows[-1]["source_order"]),
        "delta_k_min_MPa_sqrt_m": min(delta_k),
        "delta_k_max_MPa_sqrt_m": max(delta_k),
        "rate_min_m_per_cycle": min(rates),
        "rate_max_m_per_cycle": max(rates),
        **fit,
    }


def contiguous_quartiles(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    count = len(rows)
    boundaries = [round(count * index / 4) for index in range(5)]
    return [rows[boundaries[i] : boundaries[i + 1]] for i in range(4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("literature_data/fatigue_recovered_review"),
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    normalized = root / "normalized"
    audit = root / "audit"

    ca_rows = read_rows(normalized / "qste340tm_constant_amplitude.csv")
    overload_rows = read_rows(normalized / "qste340tm_overload.csv")
    history_rows = read_rows(normalized / "qste340tm_crack_histories.csv")
    closure_rows = read_rows(normalized / "qste340tm_closure_loads.csv")

    ca_by_r: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in ca_rows:
        ca_by_r[float(row["R"])].append(row)
    for rows in ca_by_r.values():
        rows.sort(key=lambda row: int(row["source_order"]))

    curve_rows: list[dict[str, object]] = []
    for r_value in sorted(ca_by_r):
        rows = ca_by_r[r_value]
        whole = summarize_segment(rows)
        curve_rows.append({"R": r_value, "segment": "full", **whole})
        for index, segment in enumerate(contiguous_quartiles(rows), start=1):
            curve_rows.append(
                {"R": r_value, "segment": f"source_order_quartile_{index}", **summarize_segment(segment)}
            )

    curve_path = audit / "qste340tm_constant_amplitude_characterization.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    blocks: list[dict[str, object]] = []
    for r_value in sorted(ca_by_r):
        blocks.append(
            {
                "block_id": f"CA_R{r_value:.1f}",
                "role": "constant_amplitude_rate_and_shape",
                "R": r_value,
                "overload_ratio": "",
                "points": len(ca_by_r[r_value]),
                "recommended_use": "one complete R block held out; rotate over four folds",
                "likelihood_independence": "one block; not independent of its matching a(N) history",
            }
        )
    overload_by_key: dict[tuple[float, float], int] = defaultdict(int)
    for row in overload_rows:
        overload_by_key[(float(row["R"]), float(row["overload_ratio_label"]))] += 1
    for (r_value, olr), points in sorted(overload_by_key.items()):
        blocks.append(
            {
                "block_id": f"OL_R{r_value:.1f}_OLR{olr:.1f}",
                "role": "paired_history_mechanism_validation",
                "R": r_value,
                "overload_ratio": olr,
                "points": points,
                "recommended_use": "hold out as a complete protocol history; no barrier refit",
                "likelihood_independence": "growth-rate, a(N), and closure views are correlated observations",
            }
        )
    block_path = audit / "qste340tm_protocol_blocks.csv"
    with block_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)

    full_curves = [row for row in curve_rows if row["segment"] == "full"]
    result = {
        "schema": "qste340tm-import-characterization/1",
        "model_fit_performed": False,
        "native_simulation_performed": False,
        "sorting_or_smoothing_performed": False,
        "constant_amplitude": {
            "points": len(ca_rows),
            "R_blocks": len(ca_by_r),
            "full_curve_loglog_slopes": {str(row["R"]): row["slope"] for row in full_curves},
            "full_curve_r_squared": {str(row["R"]): row["r_squared"] for row in full_curves},
            "quartile_slope_ranges": {
                str(r_value): [
                    min(row["slope"] for row in curve_rows if row["R"] == r_value and row["segment"] != "full"),
                    max(row["slope"] for row in curve_rows if row["R"] == r_value and row["segment"] != "full"),
                ]
                for r_value in sorted(ca_by_r)
            },
        },
        "paired_views": {
            "overload_rate_points": len(overload_rows),
            "crack_history_points": len(history_rows),
            "closure_load_points": len(closure_rows),
            "independent_sample_count_not_claimed": True,
        },
        "eligibility": {
            "data_level_shape_characterization": "ready",
            "absolute_kinetic_Gstar_calibration": "blocked_missing_protocol_and_material_local_mapping",
            "stateful_overload_validation": "blocked_missing_force_schedule_overload_timing_and_observation_map",
        },
        "outputs": [curve_path.relative_to(root).as_posix(), block_path.relative_to(root).as_posix()],
    }
    result_path = audit / "qste340tm_import_analysis.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
