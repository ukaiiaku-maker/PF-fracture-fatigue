#!/usr/bin/env python3
"""Build a signed kernel family with crack-path extension as the only state axis."""
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
from arrhenius_fracture.interaction_integral_v1029 import MODEL_ID as II_MODEL_ID
from arrhenius_fracture.signed_kernel_family_v1029 import SCHEMA as V1029_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_v10_2_6_state_resolved_kernel_family.py"
MODEL_ID = "v10.2.13_extension_only_family_with_compatibility_coordinates"
OPENING_COMPATIBILITY_COORDINATE = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--fixed-kernel-tolerance", type=float, default=0.05)
    parser.add_argument("--interpolation-neighbors", type=int, default=4)
    parser.add_argument("--reference-state-id")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.responses.is_file():
        raise SystemExit(f"response table is missing: {args.responses}")
    with args.responses.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty extension-only response table")
    schemas = {str(row.get("interaction_integral_schema", "")) for row in rows}
    if schemas != {II_MODEL_ID}:
        raise SystemExit(
            f"all responses must use {II_MODEL_ID}; incompatible schemas={sorted(schemas)}"
        )
    radii = {float(row["r_eff_over_r0"]) for row in rows}
    openings = {float(row["opening_strength_fraction"]) for row in rows}
    extensions = {float(row["crack_extension_m"]) for row in rows}
    if radii != {KERNEL_RADIUS_COMPATIBILITY_COORDINATE}:
        raise SystemExit("extension-only family requires one constant radius coordinate")
    if openings != {OPENING_COMPATIBILITY_COORDINATE}:
        raise SystemExit("extension-only family requires one constant opening coordinate")
    if len(extensions) < 2:
        raise SystemExit("extension-only family requires at least two crack-path extensions")

    with tempfile.TemporaryDirectory(prefix="v10213_extension_only_") as temp_dir:
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
            "1",
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

    gates = {
        "v10_2_6_state_coverage_passed": bool(
            payload.get("state_coverage", {}).get("coverage_passed", False)
        ),
        "single_constant_radius_compatibility_level": True,
        "single_constant_opening_compatibility_level": True,
        "multiple_cumulative_crack_path_extensions": len(extensions) >= 2,
        "analytic_interaction_integral_provenance": True,
    }
    payload.update(
        {
            "schema": V1029_SCHEMA,
            "analytic_auxiliary_gradients": True,
            "hermite_domain_weight": True,
            "interaction_integral_schema": II_MODEL_ID,
            "complete_cartesian_state_grid": True,
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "kernel_radius_compatibility_coordinate": (
                KERNEL_RADIUS_COMPATIBILITY_COORDINATE
            ),
            "opening_axis_policy": "validation_only_collapsed_constant_compatibility",
            "kernel_opening_compatibility_coordinate": (
                OPENING_COMPATIBILITY_COORDINATE
            ),
            "active_physical_kernel_axes": [
                "cumulative_crack_path_extension_m"
            ],
            "crack_extension_m_semantics": "cumulative_crack_path_extension_m",
            "opening_boundary_policy": {
                "policy": "strict",
                "lower_boundary_validated": False,
                "upper_boundary_validated": False,
            },
            "authorization_gates": gates,
            "production_parameterization_allowed": False,
            "v10_2_13_state_semantics": {
                "analytical_r_eff_used_for_interpolation": False,
                "opening_strength_fraction_used_for_interpolation": False,
                "cumulative_crack_path_extension_used_for_interpolation": True,
                "crack_extension_extrapolation_allowed": False,
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
                "extension_levels_m": sorted(extensions),
                "authorization_gates": gates,
                "production_parameterization_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
