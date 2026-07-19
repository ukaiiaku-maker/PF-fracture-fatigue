#!/usr/bin/env python3
"""Build a reviewed v10.2.13 extension-only real signed shielding atlas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from arrhenius_fracture.checked_spatial_station_projection_v10212 import (
    KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
    expand_station_response_files,
)
from arrhenius_fracture.frozen_geometry_load_invariance_v10213 import (
    MODEL_ID as LOAD_INVARIANCE_MODEL_ID,
)
from arrhenius_fracture.mechanics_normalization_v10212 import (
    MODEL_ID as NORMALIZATION_MODEL_ID,
)
from arrhenius_fracture.physical_fem_snapshot_v10212 import RESPONSE_COLUMNS
from arrhenius_fracture.signed_kernel_family_v10213 import (
    OPENING_COMPATIBILITY_COORDINATE,
    SCHEMA,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_v10_2_13_extension_only_family.py"
REVIEW_SCHEMA = "v10.2.13_independent_extension_only_real_signed_atlas_review"
REVIEW_CHECKS = (
    "physical_fem_snapshots_reviewed",
    "frozen_geometry_load_invariance_reviewed",
    "signed_and_multi_amplitude_linearity_reviewed",
    "interaction_integral_contour_stability_reviewed",
    "mesh_ribbon_and_station_convergence_reviewed",
    "cumulative_path_extension_semantics_reviewed",
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


def _load_review(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != REVIEW_SCHEMA:
        raise SystemExit(f"review schema must be {REVIEW_SCHEMA}")
    missing = [name for name in REVIEW_CHECKS if payload.get(name) is not True]
    if missing:
        raise SystemExit("independent review is incomplete: " + ", ".join(missing))
    if not str(payload.get("reviewer", "")).strip():
        raise SystemExit("independent review requires reviewer")
    if not str(payload.get("reviewed_utc", "")).strip():
        raise SystemExit("independent review requires reviewed_utc")
    return payload


def _load_invariance(paths: list[Path]) -> tuple[dict[str, dict], dict[str, str]]:
    by_parent = {}
    response_owner = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"load-invariance report is missing: {path}")
        payload = json.loads(path.read_text())
        if payload.get("schema") != LOAD_INVARIANCE_MODEL_ID:
            raise SystemExit(
                f"load-invariance report must use {LOAD_INVARIANCE_MODEL_ID}: {path}"
            )
        if payload.get("load_invariance_passed") is not True:
            raise SystemExit(f"frozen-geometry load invariance failed: {path}")
        if payload.get("opening_is_production_interpolation_axis") is not False:
            raise SystemExit("opening must be validation-only after load-invariance review")
        parent = str(payload.get("parent_state_id", "")).strip()
        if not parent or parent in by_parent:
            raise SystemExit(f"invalid or duplicate parent state in {path}")
        by_parent[parent] = payload
        found_reference = False
        for case in payload.get("generated_load_cases", []):
            if abs(float(case.get("load_scale", -1.0)) - 1.0) > 1.0e-12:
                continue
            response = str(Path(case["responses"]).expanduser().resolve())
            response_owner[response] = parent
            found_reference = True
        if not found_reference:
            raise SystemExit(f"load-invariance report has no load_scale=1 response: {path}")
    return by_parent, response_owner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--load-invariance", type=Path, action="append", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path)
    parser.add_argument("--authorize-production-parameterization", action="store_true")
    parser.add_argument("--relative-linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--fixed-kernel-tolerance", type=float, default=0.05)
    parser.add_argument("--spatial-cross-validation-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.normalization.is_file():
        raise SystemExit(f"normalization artifact is missing: {args.normalization}")
    normalization = json.loads(args.normalization.read_text())
    if normalization.get("schema") != NORMALIZATION_MODEL_ID:
        raise SystemExit(f"normalization must use {NORMALIZATION_MODEL_ID}")
    if normalization.get("normalization_source") != "process_zone_geometry_and_line_spacing":
        raise SystemExit("source capacity must come from process-zone geometry and line spacing")
    if normalization.get("fitted_to_toughness_or_fatigue") is not False:
        raise SystemExit("normalization must not be fitted to toughness or fatigue")
    if normalization.get("shielding_attenuation_factor_fitted") is not False:
        raise SystemExit("fitted shielding attenuation is prohibited")

    invariance, response_owner = _load_invariance(args.load_invariance)
    response_paths = [path.expanduser().resolve() for path in args.responses]
    missing_owner = [str(path) for path in response_paths if str(path) not in response_owner]
    if missing_owner:
        raise SystemExit(
            "each response must be the load_scale=1 station response from a supplied "
            "load-invariance report: " + ", ".join(missing_owner)
        )

    try:
        expanded_rows, physical_inputs, projection = expand_station_response_files(
            response_paths,
            relative_linearity_tolerance=args.relative_linearity_tolerance,
            spatial_cross_validation_tolerance=args.spatial_cross_validation_tolerance,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    old_to_parent = {}
    for path in response_paths:
        audit_path = path.with_suffix(".audit.json")
        audit = json.loads(audit_path.read_text())
        old_state = str(audit.get("state_id", "")).strip()
        parent = response_owner[str(path)]
        if not old_state or old_state in old_to_parent and old_to_parent[old_state] != parent:
            raise SystemExit("response state identifiers cannot be mapped uniquely")
        old_to_parent[old_state] = parent

    extension_owner = {}
    for row in expanded_rows:
        old_state = str(row["state_id"])
        if old_state not in old_to_parent:
            raise SystemExit(f"projected row has unknown response state {old_state!r}")
        parent = old_to_parent[old_state]
        extension = float(invariance[parent]["cumulative_crack_path_extension_m"])
        owner = extension_owner.setdefault(extension, parent)
        if owner != parent:
            raise SystemExit(
                "two frozen geometries have the same cumulative path extension: "
                f"{owner!r} and {parent!r}"
            )
        row["state_id"] = parent
        row["r_eff_over_r0"] = KERNEL_RADIUS_COMPATIBILITY_COORDINATE
        row["opening_strength_fraction"] = OPENING_COMPATIBILITY_COORDINATE
        row["crack_extension_m"] = extension
    extensions = sorted(extension_owner)
    if len(extensions) < 2:
        raise SystemExit("extension-only atlas requires at least two frozen geometries")

    projection.update(
        {
            "opening_axis_policy": "validation_only_collapsed_constant_compatibility",
            "kernel_opening_compatibility_coordinate": (
                OPENING_COMPATIBILITY_COORDINATE
            ),
            "active_physical_kernel_axes": [
                "cumulative_crack_path_extension_m"
            ],
            "crack_extension_m_semantics": "cumulative_crack_path_extension_m",
            "opening_strength_fraction_used_for_spatial_interpolation": False,
            "finite_radius_fem_geometry_claimed": False,
        }
    )

    review = None
    if args.independent_review is not None:
        review = _load_review(args.independent_review)
    if args.authorize_production_parameterization and review is None:
        raise SystemExit("authorization requires --independent-review")

    with tempfile.TemporaryDirectory(prefix="v10213_real_atlas_") as temp_dir:
        temp = Path(temp_dir)
        projected = temp / "projected_extension_only_responses.csv"
        with projected.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
            writer.writeheader()
            writer.writerows(expanded_rows)
        intermediate = temp / "extension_only_family.json"
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
        ]
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise SystemExit(completed.stderr + completed.stdout)
        payload = json.loads(intermediate.read_text())

    physical_audits = []
    for item in physical_inputs:
        path = Path(item["path"])
        physical_audits.append(
            {
                **item,
                "sha256": _sha256(path),
                "audit_sha256": _sha256(path.with_suffix(".audit.json")),
            }
        )
    invariance_audits = []
    for path in args.load_invariance:
        report = json.loads(path.read_text())
        invariance_audits.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "parent_state_id": report["parent_state_id"],
                "maximum_relative_load_variation": report["checks"][
                    "maximum_relative_load_variation"
                ],
            }
        )
    base_gates = dict(payload.get("authorization_gates", {}))
    real_gates = {
        "physical_fem_station_inputs_present": bool(physical_audits),
        "frozen_geometry_load_invariance_passed": all(
            item.get("load_invariance_passed") is True for item in invariance.values()
        ),
        "opening_axis_collapsed_after_validation": True,
        "multiple_cumulative_path_extensions_present": len(extensions) >= 2,
        "spatial_projection_cross_validation_passed": bool(
            projection.get("spatial_cross_validation_passed", False)
        ),
        "subelement_rows_not_claimed_as_direct_fem": bool(
            projection.get("subelement_rows_claimed_as_direct_fem") is False
        ),
        "interaction_integral_provenance_consistent": bool(
            projection.get("projected_schema_matches_measured_schema") is True
        ),
        "mechanics_derived_activation_to_line_normalization": True,
        "mechanics_derived_source_capacity_bounds": True,
        "no_fitted_shielding_attenuation": True,
        "base_extension_only_kernel_gates_passed": all(base_gates.values()),
        "independent_review_complete": review is not None,
    }
    all_gates = all(real_gates.values())
    if args.authorize_production_parameterization and not all_gates:
        raise SystemExit(
            "cannot authorize extension-only signed atlas; failed gates="
            + ",".join(key for key, passed in real_gates.items() if not passed)
        )

    payload.update(
        {
            "schema": SCHEMA,
            "physical_response_inputs": physical_audits,
            "frozen_geometry_load_invariance_inputs": invariance_audits,
            "physical_state_ids": sorted(invariance),
            "cumulative_crack_path_extension_levels_m": extensions,
            "measured_station_projection": projection,
            "normalization_path": str(args.normalization.resolve()),
            "normalization_sha256": _sha256(args.normalization),
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
            "frozen_geometry_load_invariance_passed": True,
            "full_mpz_grid_values_are_spatial_projection": True,
            "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
            "analytical_r_eff_used_for_interpolation": False,
            "opening_strength_fraction_used_for_interpolation": False,
            "finite_radius_fem_kernel_claimed": False,
            "fem_tip_geometry_blunted": False,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "states": len(invariance),
                "cumulative_path_extension_levels_m": extensions,
                "maximum_load_variation": max(
                    item["maximum_relative_load_variation"]
                    for item in invariance_audits
                ),
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
