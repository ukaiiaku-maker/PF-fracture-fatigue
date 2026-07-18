#!/usr/bin/env python3
"""Build a reviewed v10.2.9 signed shielding-kernel family.

The v10.2.6 builder remains the authoritative sign and multi-amplitude
linearity checker.  This wrapper adds the missing production gates: response
provenance from the analytic-gradient interaction integral, a complete
Cartesian state grid, physical opening coverage, and explicit lower/upper
boundary-stationarity tests before opening-axis saturation can be enabled.
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

from arrhenius_fracture.interaction_integral_v1029 import MODEL_ID as II_MODEL_ID
from arrhenius_fracture.signed_kernel_family_v1029 import SCHEMA, STATE_AXES

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_v10_2_6_state_resolved_kernel_family.py"


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
            "state atlas is not a complete Cartesian grid; interpolation across "
            "unsampled corners is prohibited"
        )
    if len(opening_levels) < 3:
        raise SystemExit("opening boundary validation requires at least three levels")

    lower_variations = []
    upper_variations = []
    for r in r_levels:
        for extension in extension_levels:
            lower_variations.append(
                _relative_pair_variation(
                    mapping[(r, opening_levels[0], extension)],
                    mapping[(r, opening_levels[1], extension)],
                    floor_fraction,
                )
            )
            upper_variations.append(
                _relative_pair_variation(
                    mapping[(r, opening_levels[-1], extension)],
                    mapping[(r, opening_levels[-2], extension)],
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
    parser.add_argument("--authorize-production-parameterization", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.responses.is_file():
        raise SystemExit(f"response table is missing: {args.responses}")

    with args.responses.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "interaction_integral_schema" not in rows[0]:
        raise SystemExit("response table lacks interaction_integral_schema provenance")
    bad = sorted(
        {
            str(row.get("interaction_integral_schema", ""))
            for row in rows
            if str(row.get("interaction_integral_schema", "")) != II_MODEL_ID
        }
    )
    if bad:
        raise SystemExit(
            f"all responses must use {II_MODEL_ID}; incompatible schemas={bad}"
        )

    with tempfile.TemporaryDirectory(prefix="v1029_kernel_") as temp_dir:
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
        "complete_cartesian_state_grid": True,
        "physical_opening_interval_covered": bool(
            boundary["physical_opening_interval_covered"]
        ),
        "lower_boundary_validated": bool(boundary["lower_boundary_validated"]),
        "upper_boundary_validated": bool(boundary["upper_boundary_validated"]),
        "analytic_interaction_integral_provenance": True,
    }
    all_gates = all(authorization_gates.values())
    if args.authorize_production_parameterization and not all_gates:
        raise SystemExit(
            "cannot authorize production parameterization; failed gates="
            + ",".join(key for key, passed in authorization_gates.items() if not passed)
        )

    payload.update(
        {
            "schema": SCHEMA,
            "analytic_auxiliary_gradients": True,
            "hermite_domain_weight": True,
            "interaction_integral_schema": II_MODEL_ID,
            "complete_cartesian_state_grid": True,
            "opening_boundary_policy": boundary,
            "authorization_gates": authorization_gates,
            "production_parameterization_allowed": bool(
                args.authorize_production_parameterization and all_gates
            ),
            "v10_2_9_hardening": {
                "effective_opening_fixed_point_required": True,
                "tip_radius_extrapolation_allowed": False,
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
                "opening_boundary_policy": boundary["policy"],
                "authorization_gates": authorization_gates,
                "production_parameterization_allowed": payload[
                    "production_parameterization_allowed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
