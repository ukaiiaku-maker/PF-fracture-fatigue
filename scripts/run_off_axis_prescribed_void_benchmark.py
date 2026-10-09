#!/usr/bin/env python3
"""Mirrored prescribed-void steering screen with frozen static mechanics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.directional_competition_v11 import tungsten_cleavage_candidates
from arrhenius_fracture.off_axis_void_steering import (
    circular_void_path_offset_m, initial_steering_prediction,
    segment_circle_clearance_m,
)

RADIUS_M = 5.5e-5
TIP_M = (4.0e-4, 0.0)
ROOT_M = (0.0, 0.0)
DELTA_A_M = 7.5e-5
OPENING_M = 4.0e-7
LEVELS = ((2.0, 1.1), (3.0, 1.3), (4.0, 1.5))


def finite(value):
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite(v) for v in value]
    if isinstance(value, np.ndarray): return finite(value.tolist())
    if isinstance(value, np.generic): return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def digest(value):
    return hashlib.sha256(json.dumps(
        finite(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def solve(path, center, *, cavity_enabled=True, width=1.0e-3, height=1.0e-3):
    return solve_crack_void_case(
        cavity_center_m=center, cavity_radius_m=RADIUS_M,
        boundary_segments=32, radial_layers=12,
        opening_m=OPENING_M*height/1.0e-3,
        crack_enabled=True, cavity_enabled=cavity_enabled,
        specimen_width_m=width, specimen_height_m=height, far_h_m=5.0e-5,
        crack_path_m=tuple(path), geometry_mode="V3_FIXED_LABORATORY_GEOMETRY",
    )


def screen_case(name, center, *, cavity_enabled=True, width=1.0e-3, height=1.0e-3):
    candidates = tungsten_cleavage_candidates(theta_deg=0.0, include_110=True)
    base = solve((ROOT_M, TIP_M), center, cavity_enabled=cavity_enabled,
                 width=width, height=height)
    base_u = float(base["observables"]["stored_energy_J_per_m"])
    rows = []
    for candidate in candidates:
        direction = np.asarray(candidate.direction_xy, dtype=float)
        endpoint = np.asarray(TIP_M)+DELTA_A_M*direction
        clearance = segment_circle_clearance_m(
            TIP_M, endpoint, cavity_center_m=center, cavity_radius_m=RADIUS_M)
        try:
            trial = solve((ROOT_M, TIP_M, tuple(endpoint)), center,
                          cavity_enabled=cavity_enabled, width=width, height=height)
        except RuntimeError as exc:
            rows.append({
                "candidate_id": candidate.candidate_id,
                "plane_variant": candidate.plane_variant,
                "angle_deg": math.degrees(candidate.angle_rad),
                "gamma_rel": candidate.gamma_rel,
                "endpoint_m": endpoint,
                "finite_segment_ideal_circle_clearance_m": clearance,
                "finite_segment_encounters_ideal_void": clearance <= 0.0,
                "status": "REJECTED_BY_UNCHANGED_V12_SUPPORT_CERTIFICATE",
                "rejection": str(exc),
                "G_finite_extension_J_per_m2": None,
                "G_over_gamma_J_per_m2": None,
            })
            continue
        obs = trial["observables"]
        G = (base_u-float(obs["stored_energy_J_per_m"]))/DELTA_A_M
        rows.append({
            "candidate_id": candidate.candidate_id,
            "plane_variant": candidate.plane_variant,
            "angle_deg": math.degrees(candidate.angle_rad),
            "gamma_rel": candidate.gamma_rel,
            "endpoint_m": endpoint,
            "finite_segment_ideal_circle_clearance_m": clearance,
            "finite_segment_encounters_ideal_void": clearance <= 0.0,
            "status": "SOLVED",
            "G_finite_extension_J_per_m2": G,
            "G_over_gamma_J_per_m2": G/candidate.gamma_rel,
            "reaction_top_N_per_m": obs["reaction_top_N_per_m"],
            "reaction_bottom_N_per_m": obs["reaction_bottom_N_per_m"],
            "compliance_m2_per_N": obs["compliance_m2_per_N"],
            "stored_energy_J_per_m": obs["stored_energy_J_per_m"],
            "weak_cavity_boundary_residual_relative": obs[
                "weak_cavity_boundary_residual_relative"],
            "mesh_minimum_quality": obs["mesh_minimum_quality"],
            "mesh_maximum_aspect_ratio": obs["mesh_maximum_aspect_ratio"],
        })
    eligible = [row for row in rows if row["status"] == "SOLVED"
                and not row["finite_segment_encounters_ideal_void"]]
    winner = (None if not eligible else
              max(eligible, key=lambda row: row["G_over_gamma_J_per_m2"]))
    straight_clearance = segment_circle_clearance_m(
        TIP_M, (TIP_M[0]+8.0*DELTA_A_M, TIP_M[1]),
        cavity_center_m=center, cavity_radius_m=RADIUS_M)
    return finite({
        "case": name, "cavity_enabled": cavity_enabled,
        "cavity_center_m": center, "radius_m": RADIUS_M,
        "specimen_width_m": width, "specimen_height_m": height,
        "opening_m": OPENING_M*height/1.0e-3,
        "outer_boundary_loading_policy": "CONSTANT_REMOTE_OPENING_STRAIN",
        "fixed_initial_crack_path_m": (ROOT_M, TIP_M),
        "straight_extension_ideal_clearance_m": straight_clearance,
        "straight_extension_misses_void": straight_clearance > 0.0,
        "base_observables": base["observables"],
        "directional_curve": rows,
        "winner_by_G_over_gamma": winner,
        "candidate_inventory": [candidate.to_dict() for candidate in candidates],
        "void_formation_suppressed": True,
        "prescribed_cavity_fixed": True,
        "path_status": "STATIC_ONE_INCREMENT_SCREEN_NOT_ACCEPTED_EVOLVING_TOPOLOGY",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT/"artifacts/off_axis_prescribed_void_benchmark")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for d_over_r, h_over_r in LEVELS:
        for sign, label in ((1.0, "above"), (-1.0, "below")):
            center = (TIP_M[0]+d_over_r*RADIUS_M, sign*h_over_r*RADIUS_M)
            case = screen_case(f"d{d_over_r:g}_h{h_over_r:g}_{label}", center)
            case["d_over_R"] = d_over_r; case["abs_h_over_R"] = h_over_r
            case["asymptotic_initial_prediction"] = initial_steering_prediction(
                TIP_M[0], cavity_center_m=center, cavity_radius_m=RADIUS_M,
                poisson_ratio=0.3)
            x = np.linspace(TIP_M[0], min(center[0]+4.0*RADIUS_M, 9.5e-4), 101)
            case["asymptotic_curve_m"] = np.column_stack((
                x, circular_void_path_offset_m(
                    x, cavity_center_m=center, cavity_radius_m=RADIUS_M,
                    poisson_ratio=0.3)))
            cases.append(case)
    central_center = (TIP_M[0]+3.0*RADIUS_M, 1.3*RADIUS_M)
    no_void = screen_case("no_void", central_center, cavity_enabled=False)
    enlarged = {
        label: screen_case(f"central_{label}_outer_enlarged", center,
                           width=1.5e-3, height=1.5e-3)
        for label, center in (
            ("above", central_center),
            ("below", (central_center[0], -central_center[1])),
        )
    }
    central = {case["case"].split("_")[-1]: case for case in cases
               if case["d_over_R"] == 3.0}
    boundary_check = {}
    for label in ("above", "below"):
        coarse = {row["plane_variant"]: row for row in central[label]["directional_curve"]
                  if row["status"] == "SOLVED"}
        large = {row["plane_variant"]: row for row in enlarged[label]["directional_curve"]
                 if row["status"] == "SOLVED"}
        boundary_check[label] = {
            key: abs(large[key]["G_finite_extension_J_per_m2"]-
                     coarse[key]["G_finite_extension_J_per_m2"])/max(
                         abs(large[key]["G_finite_extension_J_per_m2"]), 1.0e-300)
            for key in large if key in coarse
        }
    payload = finite({
        "schema": "unified-2d.off-axis-prescribed-void-benchmark/1",
        "classification": "DIAGNOSTIC_PRESCRIBED_VOID_STATIC_SCREEN",
        "physical_increment_m": DELTA_A_M,
        "crystallography": "existing {100}+optional {110}, theta=0 deg",
        "cases": cases, "no_void": no_void,
        "outer_boundary_enlargement_cases": enlarged,
        "outer_boundary_relative_G_changes": boundary_check,
        "evolving_path_gate": {
            "status": "PENDING",
            "reason": "UNCHANGED_V12_JUNCTION_SECTOR_CERTIFICATE_REJECTS_KINKED_EXTENSION",
            "certificate_was_not_weakened": True,
        },
    })
    payload["authentication_sha256"] = digest(payload)
    (args.out/"benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("OFF_AXIS_PRESCRIBED_VOID_STATIC_SCREEN = INCOMPLETE")
    for case in cases:
        winner = case["winner_by_G_over_gamma"]
        print(case["case"], case["asymptotic_initial_prediction"]["tangent_angle_deg"],
              None if winner is None else winner["plane_variant"],
              None if winner is None else winner["angle_deg"],
              None if winner is None else winner["G_finite_extension_J_per_m2"])
    print("EVOLVING_OFF_AXIS_PATH = PENDING_V12_JUNCTION_CERTIFICATE")


if __name__ == "__main__":
    main()
