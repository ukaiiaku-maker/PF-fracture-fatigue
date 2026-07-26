#!/usr/bin/env python3
"""Build a signed shielding family from direct prescribed-geometry FEM solves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import load_configuration
from arrhenius_fracture.kernel_registry_v10227 import (
    family_physics_fingerprint,
    sha256_file,
    validate_family,
)
from arrhenius_fracture.prescribed_geometry_kernel_v10228 import (
    MANIFEST_SCHEMA,
    build_prescribed_geometry_snapshots,
)

BUILD_SCHEMA = "v10.2.28_direct_prescribed_geometry_kernel_build_v1"
VALIDATION_SCHEMA = "v10.2.28_direct_kernel_validation_v1"


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"direct prescribed-geometry kernel command failed ({completed.returncode}): "
            + " ".join(command)
        )


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--family-out", type=Path, required=True)
    parser.add_argument("--required-max-extension-um", type=float, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--da-phys-um", type=float, required=True)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--margin-events", type=float, default=1.0)
    parser.add_argument("--reference-opening-strain", type=float, default=1.0e-5)
    parser.add_argument("--load-scales", type=float, nargs="+", default=(0.5, 1.0, 1.5))
    parser.add_argument(
        "--perturbation-magnitudes", type=float, nargs="+", default=(0.25, 0.50)
    )
    parser.add_argument("--linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--load-invariance-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    config_path = args.mechanical_config.expanduser().resolve()
    configuration = load_configuration(config_path)
    if configuration.branching_mode != "single_front" or configuration.maximum_fronts != 1:
        raise SystemExit("v10.2.28 direct prescribed-geometry provider is single-front only")
    if abs(float(configuration.theta_deg) - float(args.theta_deg)) > 1.0e-10:
        raise SystemExit("mechanical configuration theta does not match requested theta")

    outroot = args.outroot.expanduser().resolve()
    family_out = args.family_out.expanduser().resolve()
    if family_out.exists():
        audit = validate_family(
            family_out,
            expected_configuration_fingerprint=configuration.fingerprint(),
        )
        print(json.dumps({"reused": True, **audit}, indent=2, sort_keys=True))
        return 0
    outroot.mkdir(parents=True, exist_ok=True)
    snapshots = outroot / "prescribed_geometry_snapshots"
    loads = outroot / "load_invariance"
    if snapshots.exists() or loads.exists():
        raise SystemExit(
            "direct build scratch roots already exist; resolver must clear an invalid or "
            "incomplete cache before rebuilding"
        )

    snapshot_manifest = build_prescribed_geometry_snapshots(
        configuration,
        required_max_extension_um=float(args.required_max_extension_um),
        outroot=snapshots,
        reference_opening_strain=float(args.reference_opening_strain),
    )
    if snapshot_manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("direct snapshot manifest schema mismatch")

    loads.mkdir(parents=True)
    snapshot_paths = sorted(snapshots.glob("*/snapshot.json"))
    if len(snapshot_paths) < 2:
        raise RuntimeError("direct provider produced fewer than two geometry anchors")
    for snapshot_json in snapshot_paths:
        state_root = snapshot_json.parent
        destination = loads / state_root.name
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_v10_2_14_active_load_invariance.py"),
                "--snapshot",
                str(state_root),
                "--outroot",
                str(destination),
                "--load-scales",
                *[f"{value:.17g}" for value in args.load_scales],
                "--magnitudes",
                *[f"{value:.17g}" for value in args.perturbation_magnitudes],
                "--linearity-tolerance",
                f"{args.linearity_tolerance:.17g}",
                "--load-invariance-tolerance",
                f"{args.load_invariance_tolerance:.17g}",
                "--minimum-residual-stiffness-fraction",
                "0.001",
                "--minimum-station-spacing-m",
                f"{configuration.process_zone_length_m:.17g}",
            ]
        )

    portable_out = outroot / "portable_assembly"
    portable_out.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_v10_2_27_kernel_from_mechanics_artifacts.py"),
            "--snapshot-root",
            str(snapshots),
            "--load-invariance-root",
            str(loads),
            "--mechanical-config",
            str(config_path),
            "--outroot",
            str(portable_out),
            "--family-out",
            str(family_out),
            "--target-extension-um",
            f"{args.target_extension_um:.17g}",
            "--theta-deg",
            f"{args.theta_deg:.17g}",
            "--da-phys-um",
            f"{args.da_phys_um:.17g}",
            "--event-minimum-factor",
            f"{args.event_minimum_factor:.17g}",
            "--event-maximum-factor",
            f"{args.event_maximum_factor:.17g}",
            "--margin-events",
            f"{args.margin_events:.17g}",
        ]
    )

    payload = _load(family_out)
    payload.update(
        {
            "mechanical_configuration": configuration.canonical_payload(),
            "mechanical_configuration_fingerprint": configuration.fingerprint(),
            "kernel_provider_id": "v10.2.28_direct_prescribed_geometry_fem_v1",
            "kernel_recalculated_from_current_configuration": True,
            "direct_prescribed_geometry": True,
            "historical_reference_condition_required": False,
            "prior_kernel_family_required": False,
            "material_parameter_option_required": False,
            "hazard_seed_required": False,
            "stochastic_trajectory_required": False,
            "production_moving_process_zone_physics_preserved": True,
            "production_physics_modified": False,
            "capture_endpoint_reconstruction_is_measurement_only": False,
            "geometry_anchors_are_prescribed_not_captured": True,
        }
    )
    family_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    audit = validate_family(
        family_out,
        expected_configuration_fingerprint=configuration.fingerprint(),
    )

    load_reports = sorted(loads.glob("*/frozen_geometry_load_invariance.json"))
    load_payloads = [_load(path) for path in load_reports]
    load_passed = bool(load_payloads) and all(
        report.get("load_invariance_passed") is True for report in load_payloads
    )
    sign_linearity_passed = bool(load_payloads) and all(
        report.get("checks", {}).get("within_load_sign_amplitude_linearity_passed") is True
        for report in load_payloads
    )
    validation = {
        "schema": VALIDATION_SCHEMA,
        "passed": bool(load_passed and sign_linearity_passed),
        "configuration_fingerprint": configuration.fingerprint(),
        "family": audit["family"],
        "family_sha256": audit["file_sha256"],
        "family_physics_fingerprint": audit["physics_fingerprint"],
        "direct_prescribed_geometry": True,
        "material_option_independent_by_construction": True,
        "temperature_independent_when_elasticity_fixed": bool(
            not configuration.temperature_dependent_mechanics
        ),
        "hazard_seed_independent_by_construction": True,
        "prior_family_independent_by_construction": True,
        "stochastic_event_independent_by_construction": True,
        "fracture_hazard_imported_or_invoked": False,
        "source_emission_imported_or_invoked": False,
        "moving_process_zone_imported_or_invoked": False,
        "load_invariance_passed": load_passed,
        "positive_negative_multi_amplitude_linearity_passed": sign_linearity_passed,
        "state_count": len(snapshot_paths),
        "load_invariance_reports": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "maximum_relative_load_variation": report.get("checks", {}).get(
                    "maximum_relative_load_variation"
                ),
                "maximum_within_load_relative_spread": report.get("checks", {}).get(
                    "maximum_within_load_relative_spread"
                ),
            }
            for path, report in zip(load_reports, load_payloads)
        ],
    }
    if not validation["passed"]:
        raise RuntimeError("direct kernel validation gates did not pass")
    validation_path = outroot / "direct_kernel_validation_manifest.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    portable_manifest_path = portable_out / "kernel_build_manifest.json"
    portable_manifest = (
        _load(portable_manifest_path) if portable_manifest_path.is_file() else {}
    )
    build_manifest = {
        **portable_manifest,
        "schema": BUILD_SCHEMA,
        "configuration": configuration.canonical_payload(),
        "configuration_fingerprint": configuration.fingerprint(),
        "family": audit["family"],
        "family_sha256": audit["file_sha256"],
        "family_physics_fingerprint": family_physics_fingerprint(family_out),
        "prescribed_geometry_snapshot_manifest": str(
            (snapshots / "kernel_capture_manifest.json").resolve()
        ),
        "prescribed_geometry_snapshot_manifest_sha256": sha256_file(
            snapshots / "kernel_capture_manifest.json"
        ),
        "direct_kernel_validation_manifest": str(validation_path.resolve()),
        "direct_kernel_validation_manifest_sha256": sha256_file(validation_path),
        "direct_provider_validated": True,
        "production_parameterization_promotion_allowed": True,
        "historical_reference_condition_required": False,
        "prior_kernel_family_required": False,
        "stochastic_trajectory_required": False,
        "production_physics_modified": False,
    }
    build_manifest_path = outroot / "kernel_build_manifest.json"
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps({"reused": False, **build_manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
