#!/usr/bin/env python3
"""Build the v10.2.12 family with a constant radius compatibility coordinate.

The existing state schema retains ``r_eff_over_r0`` for backward compatibility,
but every projected row must use one. The physical kernel axes are local opening
fraction and crack extension. No second radius level is fabricated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from arrhenius_fracture.checked_spatial_station_projection_v10212 import (
    KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
)
from scripts.build_v10_2_9_state_resolved_kernel_family import (
    II_MODEL_ID,
    SCHEMA,
    _boundary_assessment,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_v10_2_6_state_resolved_kernel_family.py"
MODEL_ID = "v10.2.12_opening_extension_family_with_constant_radius_coordinate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--fixed-kernel-tolerance", type=float, default=0.05)
    parser.add_argument("--boundary-stationarity-tolerance", type=float, default=0.05)
    parser.add_argument("--boundary-significance-floor-fraction", type=float, default=1.0e-3)
    parser.add_argument("--reference-state-id")
    parser.add_argument("--interpolation-neighbors", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.responses.is_file():
        raise SystemExit(f"response table is missing: {args.responses}")
    with args.responses.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty projected response table")
    schemas = {str(row.get("interaction_integral_schema", "")) for row in rows}
    if schemas != {II_MODEL_ID}:
        raise SystemExit(
            f"all responses must use {II_MODEL_ID}; incompatible schemas={sorted(schemas)}"
        )
    radii = {float(row["r_eff_over_r0"]) for row in rows}
    if radii != {KERNEL_RADIUS_COMPATIBILITY_COORDINATE}:
        raise SystemExit(
            "v10.2.12 projected responses must hold r_eff_over_r0 at the constant "
            f"compatibility coordinate {KERNEL_RADIUS_COMPATIBILITY_COORDINATE}; got {sorted(radii)}"
        )

    with tempfile.TemporaryDirectory(prefix="v10212_compat_radius_") as temp_dir:
        intermediate = Path(temp_dir) / "v1026_intermediate.json"
        command = [
            sys.executable,
            str(BASE_BUILDER),
            "--responses",
            str(args.responses),
            "--normalization",
            str(args.normalization),
            "--out",
            str(intermediate),
            "--relative-linearity-tolerance",
            str(args.relative_linearity_tolerance),
            "--fixed-kernel-tolerance",
            str(args.fixed_kernel_tolerance),
            "--interpolation-neighbors",
            str(args.interpolation_neighbors),
            "--minimum-distinct-r",
            "1",
            "--minimum-distinct-opening",
            "3",
            "--minimum-distinct-extension",
            "2",
        ]
        if args.reference_state_id:
            command.extend(["--reference-state-id", args.reference_state_id])
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise SystemExit(completed.stderr + completed.stdout)
        payload = json.loads(intermediate.read_text())

    boundary = _boundary_assessment(
        payload,
        tolerance=float(args.boundary_stationarity_tolerance),
        floor_fraction=float(args.boundary_significance_floor_fraction),
    )
    authorization_gates = {
        "v10_2_6_state_coverage_passed": bool(
            payload.get("state_coverage", {}).get("coverage_passed", False)
        ),
        "single_constant_radius_compatibility_level": True,
        "complete_cartesian_opening_extension_grid": True,
        "physical_opening_interval_covered": bool(
            boundary["physical_opening_interval_covered"]
        ),
        "lower_boundary_validated": bool(boundary["lower_boundary_validated"]),
        "upper_boundary_validated": bool(boundary["upper_boundary_validated"]),
        "analytic_interaction_integral_provenance": True,
    }
    payload.update(
        {
            "schema": SCHEMA,
            "analytic_auxiliary_gradients": True,
            "hermite_domain_weight": True,
            "interaction_integral_schema": II_MODEL_ID,
            "complete_cartesian_state_grid": True,
            "complete_cartesian_opening_extension_grid": True,
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "kernel_radius_compatibility_coordinate": KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
            "active_physical_kernel_axes": [
                "opening_strength_fraction",
                "crack_extension_m",
            ],
            "opening_boundary_policy": boundary,
            "authorization_gates": authorization_gates,
            "production_parameterization_allowed": False,
            "v10_2_12_state_semantics": {
                "analytical_r_eff_used_for_interpolation": False,
                "finite_radius_fem_geometry_claimed": False,
                "tip_radius_extrapolation_needed": False,
                "crack_extension_extrapolation_allowed": False,
                "opening_saturation_requires_boundary_stationarity": True,
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_states": len(payload["states"]),
                "radius_levels": [KERNEL_RADIUS_COMPATIBILITY_COORDINATE],
                "opening_boundary_policy": boundary["policy"],
                "authorization_gates": authorization_gates,
                "production_parameterization_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
