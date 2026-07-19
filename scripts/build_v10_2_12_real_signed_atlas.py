#!/usr/bin/env python3
"""Build a reviewed v10.2.12 real signed 2-D shielding-kernel atlas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from arrhenius_fracture.mechanics_normalization_v10212 import MODEL_ID as NORMALIZATION_MODEL_ID
from arrhenius_fracture.physical_fem_snapshot_v10212 import RESPONSE_COLUMNS
from arrhenius_fracture.spatial_station_projection_v10212 import (
    expand_station_response_files,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_v10_2_9_state_resolved_kernel_family.py"
MODEL_ID = "v10.2.12_real_signed_state_resolved_2d_shielding_atlas"
REVIEW_SCHEMA = "v10.2.12_independent_real_signed_atlas_review"
REVIEW_CHECKS = (
    "physical_fem_snapshots_reviewed",
    "signed_linearity_reviewed",
    "multi_amplitude_linearity_reviewed",
    "interaction_integral_contour_stability_reviewed",
    "mesh_and_ribbon_convergence_reviewed",
    "state_envelope_reviewed",
    "analytical_r_eff_axis_interpretation_reviewed",
    "source_normalization_reviewed",
    "exact_replay_reviewed",
    "full_2d_fracture_and_fatigue_validation_reviewed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _review(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"independent review is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema") != REVIEW_SCHEMA:
        raise SystemExit(f"review schema must be {REVIEW_SCHEMA}")
    missing = [name for name in REVIEW_CHECKS if payload.get(name) is not True]
    if missing:
        raise SystemExit("independent review is incomplete: " + ", ".join(missing))
    if not str(payload.get("reviewer", "")).strip() or not str(payload.get("reviewed_utc", "")).strip():
        raise SystemExit("independent review requires reviewer and reviewed_utc")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path)
    parser.add_argument("--authorize-production-parameterization", action="store_true")
    parser.add_argument("--relative-linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--fixed-kernel-tolerance", type=float, default=0.05)
    parser.add_argument("--boundary-stationarity-tolerance", type=float, default=0.05)
    parser.add_argument("--spatial-cross-validation-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.normalization.is_file():
        raise SystemExit(f"normalization artifact is missing: {args.normalization}")
    normalization = json.loads(args.normalization.read_text())
    if normalization.get("schema") != NORMALIZATION_MODEL_ID:
        raise SystemExit(
            f"normalization must use {NORMALIZATION_MODEL_ID}; got {normalization.get('schema')!r}"
        )
    if normalization.get("normalization_source") != "process_zone_geometry_and_line_spacing":
        raise SystemExit("source capacity must be derived from process-zone geometry and line spacing")
    if normalization.get("fitted_to_toughness_or_fatigue") is not False:
        raise SystemExit("normalization must not be fitted to toughness or fatigue")
    if normalization.get("shielding_attenuation_factor_fitted") is not False:
        raise SystemExit("a fitted shielding attenuation factor is prohibited")

    try:
        expanded_rows, physical_inputs, projection = expand_station_response_files(
            args.responses,
            relative_linearity_tolerance=args.relative_linearity_tolerance,
            spatial_cross_validation_tolerance=args.spatial_cross_validation_tolerance,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not expanded_rows:
        raise SystemExit("no projected MPZ-grid rows were produced")
    audits = []
    for item in physical_inputs:
        path = Path(item["path"])
        audits.append(
            {
                **item,
                "sha256": _sha256(path),
                "audit_sha256": _sha256(path.with_suffix(".audit.json")),
            }
        )
    state_ids = sorted({str(row["state_id"]) for row in expanded_rows})

    review = None
    if args.authorize_production_parameterization:
        if args.independent_review is None:
            raise SystemExit("authorization requires --independent-review")
        review = _review(args.independent_review)
    elif args.independent_review is not None:
        review = _review(args.independent_review)

    with tempfile.TemporaryDirectory(prefix="v10212_atlas_") as temp_dir:
        temp = Path(temp_dir)
        projected = temp / "projected_full_mpz_grid_responses.csv"
        with projected.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
            writer.writeheader()
            writer.writerows(expanded_rows)
        intermediate = temp / "v1029_intermediate.json"
        command = [
            sys.executable,
            str(BASE_BUILDER),
            "--responses",
            str(projected),
            "--normalization",
            str(args.normalization),
            "--out",
            str(intermediate),
            "--relative-linearity-tolerance",
            str(args.relative_linearity_tolerance),
            "--fixed-kernel-tolerance",
            str(args.fixed_kernel_tolerance),
            "--boundary-stationarity-tolerance",
            str(args.boundary_stationarity_tolerance),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.stderr + completed.stdout)
        payload = json.loads(intermediate.read_text())

    base_gates = dict(payload.get("authorization_gates", {}))
    real_gates = {
        "physical_fem_station_inputs_present": bool(audits),
        "measured_stations_distinguished_from_projected_grid": all(
            item["audit"].get("responses_are_measured_stations_not_full_grid") is True
            for item in audits
        ),
        "physical_signed_and_multi_amplitude_linearity_passed": bool(
            projection.get("physical_linearity_checks")
        ),
        "spatial_projection_cross_validation_passed": bool(
            projection.get("spatial_cross_validation_passed", False)
        ),
        "subelement_rows_not_claimed_as_direct_fem": bool(
            projection.get("subelement_rows_claimed_as_direct_fem") is False
        ),
        "mechanics_derived_activation_to_line_normalization": True,
        "mechanics_derived_source_capacity_bounds": True,
        "no_fitted_shielding_attenuation": True,
        "r_eff_axis_declared_analytical_not_fem_geometry": all(
            item["audit"].get("r_eff_is_analytical_tip_state") is True for item in audits
        ),
        "base_v10_2_9_kernel_gates_passed": all(base_gates.values()),
        "independent_review_complete": review is not None,
    }
    all_gates = all(real_gates.values())
    if args.authorize_production_parameterization and not all_gates:
        raise SystemExit(
            "cannot authorize real signed atlas; failed gates="
            + ",".join(key for key, value in real_gates.items() if not value)
        )

    payload.update(
        {
            "schema": MODEL_ID,
            "physical_response_inputs": audits,
            "physical_state_ids": state_ids,
            "measured_station_projection": projection,
            "normalization_path": str(args.normalization.resolve()),
            "normalization_sha256": _sha256(args.normalization),
            "normalization_schema": NORMALIZATION_MODEL_ID,
            "independent_review": review,
            "real_atlas_authorization_gates": real_gates,
            "production_parameterization_allowed": bool(
                args.authorize_production_parameterization and all_gates
            ),
            "same_kernel_family_for_monotonic_and_fatigue": True,
            "constitutive_K_shield_cap_present": False,
            "signed_burgers_population_required": True,
            "source_sites_are_nucleation_opportunities": True,
            "population_units": "signed_dislocation_line_content",
            "fem_tip_geometry_blunted": False,
            "r_eff_axis_interpretation": (
                "analytical local-tip stress/blunting state sampled during the physical FEM run; "
                "not a finite-radius FEM crack geometry"
            ),
            "finite_radius_fem_kernel_claimed": False,
            "cohesive_network_snapshots_supported": False,
            "full_mpz_grid_values_are_spatial_projection": True,
            "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "states": len(state_ids),
                "physical_measured_rows": projection["physical_measured_row_count"],
                "projected_full_grid_rows": projection["projected_full_grid_row_count"],
                "spatial_cross_validation_passed": projection[
                    "spatial_cross_validation_passed"
                ],
                "base_gates": base_gates,
                "real_atlas_gates": real_gates,
                "production_parameterization_allowed": payload[
                    "production_parameterization_allowed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
