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
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from arrhenius_fracture.checked_spatial_station_projection_v10212 import (
    KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
)
from arrhenius_fracture.interaction_integral_v1029 import MODEL_ID as II_MODEL_ID
from arrhenius_fracture.signed_kernel_family_v1029 import SCHEMA, STATE_AXES

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_v10_2_6_state_resolved_kernel_family.py"
MODEL_ID = "v10.2.12_opening_extension_family_with_constant_radius_coordinate"


def _levels(states: list[dict[str, Any]], axis: str) -> list[float]:
    return sorted({float(state[axis]) for state in states})


def _state_map(states: list[dict[str, Any]]) -> dict[tuple[float, float, float], dict[str, Any]]:
    return {
        tuple(float(state[name]) for name in STATE_AXES): state for state in states
    }


def _kernel_arrays(state: dict[str, Any]):
    for key in (
        "active_kernel_I_Pa_sqrt_m_per_signed_line",
        "wake_kernel_I_Pa_sqrt_m_per_signed_line",
        "active_kernel_II_Pa_sqrt_m_per_signed_line",
        "wake_kernel_II_Pa_sqrt_m_per_signed_line",
    ):
        yield key, np.asarray(state.get(key, []), dtype=float)


def _relative_pair_variation(
    boundary: dict[str, Any], interior: dict[str, Any], floor_fraction: float
) -> float:
    worst = 0.0
    for (name_a, a), (name_b, b) in zip(
        _kernel_arrays(boundary), _kernel_arrays(interior)
    ):
        if name_a != name_b or a.shape != b.shape:
            raise SystemExit("boundary-state kernel shapes are inconsistent")
        if a.size == 0:
            continue
        floor = max(float(np.max(np.abs(a))) * floor_fraction, 1.0e-12)
        relative = np.abs(b - a) / np.maximum(np.abs(a), floor)
        worst = max(worst, float(np.max(relative)))
    return worst


def _boundary_assessment(
    payload: dict[str, Any], *, tolerance: float, floor_fraction: float
) -> dict[str, Any]:
    states = list(payload["states"])
    r_levels = _levels(states, "r_eff_over_r0")
    opening_levels = _levels(states, "opening_strength_fraction")
    extension_levels = _levels(states, "crack_extension_m")
    expected = len(r_levels) * len(opening_levels) * len(extension_levels)
    mapping = _state_map(states)
    complete = len(states) == expected and len(mapping) == expected
    if complete:
        for r in r_levels:
            for opening in opening_levels:
                for extension in extension_levels:
                    complete = complete and (r, opening, extension) in mapping
    if not complete:
        raise SystemExit(
            "state atlas is not a complete Cartesian opening-extension grid"
        )
    if len(r_levels) != 1 or not math.isclose(
        r_levels[0],
        KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise SystemExit("compatibility-radius family requires exactly one radius level")
    if len(opening_levels) < 3:
        raise SystemExit("opening boundary validation requires at least three levels")

    lower_variations = []
    upper_variations = []
    for extension in extension_levels:
        radius = r_levels[0]
        lower_variations.append(
            _relative_pair_variation(
                mapping[(radius, opening_levels[0], extension)],
                mapping[(radius, opening_levels[1], extension)],
                floor_fraction,
            )
        )
        upper_variations.append(
            _relative_pair_variation(
                mapping[(radius, opening_levels[-1], extension)],
                mapping[(radius, opening_levels[-2], extension)],
                floor_fraction,
            )
        )
    lower_worst = max(lower_variations, default=math.inf)
    upper_worst = max(upper_variations, default=math.inf)
    physical_coverage = bool(opening_levels[0] <= 0.05 and opening_levels[-1] >= 0.95)
    return {
        "policy": (
            "validated_boundary_saturation"
            if physical_coverage and lower_worst <= tolerance and upper_worst <= tolerance
            else "strict"
        ),
        "opening_levels": opening_levels,
        "physical_opening_interval_covered": physical_coverage,
        "lower_boundary_max_relative_change_to_next_level": lower_worst,
        "upper_boundary_max_relative_change_to_next_level": upper_worst,
        "boundary_stationarity_tolerance": float(tolerance),
        "lower_boundary_validated": bool(
            physical_coverage and lower_worst <= tolerance
        ),
        "upper_boundary_validated": bool(
            physical_coverage and upper_worst <= tolerance
        ),
        "complete_cartesian_state_grid": True,
        "complete_cartesian_opening_extension_grid": True,
        "n_expected_states": expected,
        "n_actual_states": len(states),
    }


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
