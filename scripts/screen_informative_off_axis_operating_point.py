#!/usr/bin/env python3
"""Prospective active-row placement screen for the off-axis hole experiment."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_off_axis_prescribed_void_benchmark_v2 import (
    RADIUS, TIP, finite, solve_case,
)

MATERIAL = "ceramic-like"
TEMPERATURE_K = 300.0
LOAD_SCALE = 9.5
PLACEMENTS = (
    ("near_high_slope", 2.2, 1.4),
    ("near_low_offset", 2.5, 1.1),
    ("far_high_offset", 3.0, 1.8),
)


def centers(d_over_r: float, h_over_r: float):
    x = TIP[0] + d_over_r*RADIUS
    y = h_over_r*RADIUS
    return {"above": (x, y), "below": (x, -y)}


def dump(path: Path, value):
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True)+"\n")


def load(path: Path):
    return json.loads(path.read_text())


def main():
    out = ROOT/"artifacts/informative_off_axis_active_screen"
    out.mkdir(parents=True, exist_ok=True)
    no_void_path = out/"no_void.json"
    if no_void_path.exists():
        no_void = load(no_void_path)
    else:
        no_void = solve_case(
            "no_void", run_scheduler=False, include_finite_diagnostic=False,
            material_family=MATERIAL, temperature_K=TEMPERATURE_K,
            load_scale=LOAD_SCALE, maximum_observation_s=1.0e-3,
        )
        dump(no_void_path, no_void)
    base = {row["plane_variant"]: row
            for row in no_void["canonical_kinetic_directional_curve"]}
    summaries = []
    for name, d_over_r, h_over_r in PLACEMENTS:
        result_path = out/f"{name}.json"
        if result_path.exists():
            result = load(result_path)
        else:
            result = solve_case(
                "above", run_scheduler=False, include_finite_diagnostic=False,
                material_family=MATERIAL, temperature_K=TEMPERATURE_K,
                load_scale=LOAD_SCALE, cavity_centers=centers(d_over_r, h_over_r),
                maximum_observation_s=1.0e-3,
            )
            dump(result_path, result)
        rows = {row["plane_variant"]: row
                for row in result["canonical_kinetic_directional_curve"]}
        summary = {
            "name": name, "d_over_R": d_over_r, "h_over_R": h_over_r,
            "asymptotic_prediction": result["asymptotic_prediction"],
            "straight_clearance_m": h_over_r*RADIUS-RADIUS,
            "rows": {},
        }
        for variant, row in rows.items():
            value = row["finite_increment_G_J_per_m2"]
            reference = base[variant]["finite_increment_G_J_per_m2"]
            valid = value is not None and reference is not None
            summary["rows"][variant] = {
                "G_J_per_m2": value,
                "void_delta_G_J_per_m2": value-reference if valid else None,
                "void_relative_change": ((value-reference)/abs(reference)
                                         if valid else None),
                "mesh_converged": row["mesh_converged"],
            }
        summaries.append(summary)
        dump(out/"screen_progress.json", {
            "material_family": MATERIAL, "temperature_K": TEMPERATURE_K,
            "load_scale": LOAD_SCALE, "completed": summaries,
        })
        print("PLACEMENT_COMPLETE", name, flush=True)
    dump(out/"screen.json", {
        "material_family": MATERIAL, "temperature_K": TEMPERATURE_K,
        "load_scale": LOAD_SCALE, "no_void": no_void,
        "placements": summaries,
        "selection_policy": (
            "straight_misses; all rows converged; toward (+45) void effect "
            "exceeds away (-45), then largest toward-minus-away effect"
        ),
    })


if __name__ == "__main__":
    main()
