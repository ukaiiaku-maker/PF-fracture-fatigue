#!/usr/bin/env python3
"""Matched-load enlarged-domain check for the selected active interaction case."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_off_axis_prescribed_void_benchmark_v2 import finite, solve_case
from scripts.screen_informative_off_axis_operating_point import (
    LOAD_SCALE, MATERIAL, TEMPERATURE_K, centers,
)

OUT = ROOT/"artifacts/informative_off_axis_domain_sensitivity_1p4mm"
CENTRAL = ROOT/"artifacts/informative_off_axis_active_screen"
SELECTED_CENTERS = centers(2.2, 1.4)
WIDTH = HEIGHT = 1.4e-3
FAR_H = 5.0e-5
LIMIT = 0.10


def dump(path: Path, value):
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True)+"\n")


def load(path: Path):
    return json.loads(path.read_text())


def row_map(case):
    return {row["plane_variant"]: row
            for row in case["canonical_kinetic_directional_curve"]}


def solve_saved(path: Path, case: str, load_scale: float):
    if path.exists():
        return load(path)
    result = solve_case(
        case, width=WIDTH, height=HEIGHT, far_h=FAR_H,
        run_scheduler=False, include_finite_diagnostic=False,
        material_family=MATERIAL, temperature_K=TEMPERATURE_K,
        load_scale=load_scale, cavity_centers=SELECTED_CENTERS,
        maximum_observation_s=1.0e-3,
    )
    dump(path, result)
    print("DOMAIN_CASE_COMPLETE", path.stem, flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    central_no_void = load(CENTRAL/"no_void.json")
    central_above = load(CENTRAL/"near_high_slope.json")
    central_base = row_map(central_no_void)
    central_void = row_map(central_above)

    pilot = solve_saved(OUT/"outer_no_void_pilot.json", "no_void", LOAD_SCALE)
    target = central_base["(100)"]["finite_increment_G_J_per_m2"]
    pilot_G = row_map(pilot)["(100)"]["finite_increment_G_J_per_m2"]
    calibrated_scale = LOAD_SCALE*math.sqrt(target/pilot_G)
    (OUT/"calibrated_load_scale.txt").write_text(
        f"{calibrated_scale:.17g}\n")

    outer_no_void = solve_saved(
        OUT/"outer_no_void_matched.json", "no_void", calibrated_scale)
    outer_above = solve_saved(
        OUT/"outer_above_matched.json", "above", calibrated_scale)
    outer_base = row_map(outer_no_void)
    outer_void = row_map(outer_above)

    rows = []
    for variant in ("(1-10)", "(100)", "(110)"):
        cb = central_base[variant]["finite_increment_G_J_per_m2"]
        cv = central_void[variant]["finite_increment_G_J_per_m2"]
        ob = outer_base[variant]["finite_increment_G_J_per_m2"]
        ov = outer_void[variant]["finite_increment_G_J_per_m2"]
        values_available = all(value is not None for value in (cb, cv, ob, ov))
        central_delta = cv-cb if values_available else None
        outer_delta = ov-ob if values_available else None
        relative = (abs(outer_delta-central_delta)/max(
            abs(outer_delta), abs(central_delta), 1.0e-300)
            if values_available else None)
        rows.append({
            "plane_variant": variant,
            "central_no_void_G_J_per_m2": cb,
            "central_void_G_J_per_m2": cv,
            "central_void_delta_G_J_per_m2": central_delta,
            "central_normalized_void_delta": (
                central_delta/abs(cb) if values_available else None),
            "outer_no_void_G_J_per_m2": ob,
            "outer_void_G_J_per_m2": ov,
            "outer_void_delta_G_J_per_m2": outer_delta,
            "outer_normalized_void_delta": (
                outer_delta/abs(ob) if values_available else None),
            "void_delta_domain_relative_change": relative,
            "central_mesh_converged": bool(central_void[variant]["mesh_converged"]),
            "outer_mesh_converged": bool(outer_void[variant]["mesh_converged"]),
        })
    finite_changes = [row["void_delta_domain_relative_change"] for row in rows
                      if row["void_delta_domain_relative_change"] is not None]
    matched_reference_error = abs(
        row_map(outer_no_void)["(100)"]["finite_increment_G_J_per_m2"]-target)/abs(target)
    payload = {
        "selected_geometry": {"d_over_R": 2.2, "h_over_R": 1.4,
                              "straight_clearance_m": 2.2e-5},
        "material_family": MATERIAL, "temperature_K": TEMPERATURE_K,
        "central_load_scale": LOAD_SCALE,
        "outer_load_scale": calibrated_scale,
        "matching_quantity": "NO_VOID_STRAIGHT_CANONICAL_G",
        "matched_reference_relative_error": matched_reference_error,
        "graded_remote_mesh": {"width_m": WIDTH, "height_m": HEIGHT,
                               "far_h_m": FAR_H},
        "rows": rows,
        "relative_limit": LIMIT,
        "maximum_void_delta_domain_relative_change": (
            max(finite_changes) if finite_changes else None),
        "status": ("PASS" if finite_changes and max(finite_changes) <= LIMIT
                   and matched_reference_error <= 0.02
                   and all(row["central_mesh_converged"] and row["outer_mesh_converged"]
                           for row in rows)
                   else "COMPLETED_SENSITIVITY_NOT_CONVERGED"),
    }
    dump(OUT/"domain_sensitivity.json", payload)
    print("DOMAIN_SENSITIVITY", payload["status"], flush=True)


if __name__ == "__main__":
    main()
