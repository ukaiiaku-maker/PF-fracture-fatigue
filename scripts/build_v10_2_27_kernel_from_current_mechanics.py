#!/usr/bin/env python3
"""Validate current mechanics artifacts, build the kernel, and bind provenance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
    load_configuration,
)
from arrhenius_fracture.kernel_registry_v10227 import validate_family

BASE_PATH = ROOT / "scripts" / "build_v10_2_27_kernel_from_mechanics_artifacts.py"
SPEC = importlib.util.spec_from_file_location("v10227_portable_builder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load mechanics builder from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SELF_CONSISTENCY_SCHEMA = "v10.2.27_kernel_self_consistency_selection_v2"


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_self_consistency_selection(
    snapshot_root: Path,
    configuration: MechanicalKernelConfiguration,
    *,
    allow_unconverged_capture: bool,
) -> dict[str, Any]:
    """Require portable evidence of at least two converging target-family passes."""
    selection_path = snapshot_root / "kernel_self_consistency_selection.json"
    if allow_unconverged_capture:
        if selection_path.exists():
            raise ValueError(
                "provisional iteration capture unexpectedly contains a final "
                "kernel_self_consistency_selection.json"
            )
        return {
            "schema": "v10.2.27_provisional_kernel_iteration_v1",
            "validated": False,
            "allow_unconverged_capture": True,
            "selection": None,
        }
    if not selection_path.is_file():
        raise ValueError(
            "accepted production snapshots lack kernel_self_consistency_selection.json. "
            "A single capture/build pass cannot be promoted; run the bounded "
            "production-capture fixed-point workflow."
        )
    payload = json.loads(selection_path.read_text())
    failures: list[str] = []
    if payload.get("schema") != SELF_CONSISTENCY_SCHEMA:
        failures.append("schema")
    if payload.get("converged") is not True:
        failures.append("converged")
    if int(payload.get("minimum_target_family_passes", 0)) < 2:
        failures.append("minimum_target_family_passes")
    if int(payload.get("converged_iteration", -1)) < 1:
        failures.append("converged_iteration")
    expected_fingerprint = configuration.fingerprint()
    if payload.get("mechanical_configuration_fingerprint") != expected_fingerprint:
        failures.append("mechanical_configuration_fingerprint")
    candidate_sha = payload.get("converged_candidate_family_sha256")
    if not _is_sha256(candidate_sha):
        failures.append("converged_candidate_family_sha256")
    candidate_physics = payload.get(
        "converged_candidate_family_physics_fingerprint"
    )
    if not _is_sha256(candidate_physics):
        failures.append("converged_candidate_family_physics_fingerprint")
    bootstrap_sha = payload.get("initial_bootstrap_family_sha256")
    if not _is_sha256(bootstrap_sha):
        failures.append("initial_bootstrap_family_sha256")

    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        failures.append("comparisons")
        comparisons = []
    else:
        for index, comparison in enumerate(comparisons):
            if not isinstance(comparison, dict):
                failures.append(f"comparisons_{index}_type")
                continue
            if comparison.get("schema") != (
                "v10.2.27_kernel_self_consistency_comparison_v1"
            ):
                failures.append(f"comparisons_{index}_schema")
            for name in (
                "previous_family_sha256",
                "current_family_sha256",
                "previous_family_physics_fingerprint",
                "current_family_physics_fingerprint",
            ):
                if not _is_sha256(comparison.get(name)):
                    failures.append(f"comparisons_{index}_{name}")
        final = comparisons[-1] if comparisons else {}
        if final.get("converged") is not True:
            failures.append("final_comparison_converged")
        if final.get("current_family_sha256") != candidate_sha:
            failures.append("final_comparison_candidate_sha256")
        if final.get("current_family_physics_fingerprint") != candidate_physics:
            failures.append("final_comparison_candidate_physics_fingerprint")

    if failures:
        raise ValueError(
            "kernel self-consistency selection is invalid: "
            + ", ".join(sorted(set(failures)))
            + f"; selection={selection_path}"
        )
    return {
        "schema": SELF_CONSISTENCY_SCHEMA,
        "validated": True,
        "allow_unconverged_capture": False,
        "selection_path": str(selection_path),
        "selection": payload,
    }


def _validate_artifacts(
    snapshot_root: Path,
    states: list[dict[str, Any]],
    configuration: MechanicalKernelConfiguration,
) -> None:
    manifest_path = snapshot_root / "kernel_capture_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "snapshot mechanics lack a v10.2.27 configuration manifest. "
            "Legacy archives cannot be assumed compatible; recalculate from the requested "
            "mechanical configuration instead."
        )
    manifest = json.loads(manifest_path.read_text())
    expected_fingerprint = configuration.fingerprint()
    observed_fingerprint = str(
        manifest.get("mechanical_configuration_fingerprint", "")
    )
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(
            "snapshot mechanical-configuration fingerprint mismatch: "
            f"{observed_fingerprint or '<missing>'} != {expected_fingerprint}"
        )
    if manifest.get("mechanical_configuration") != configuration.canonical_payload():
        raise ValueError("snapshot mechanical-configuration payload does not match request")
    measurement_manifest = dict(manifest.get("measurement_snapshot", {}))
    for key in (
        "capture_only",
        "trajectory_state_cloned",
        "plasticity_frozen",
        "kinetics_not_advanced",
        "endpoint_mesh_re_equilibrated",
    ):
        if measurement_manifest.get(key) is not True:
            raise ValueError(f"capture manifest lacks required measurement invariant {key}")

    for row in states:
        metadata = json.loads(Path(row["snapshot_json"]).read_text())
        engine = row["engine_config"]
        front = dict(engine.get("front_config", {}))
        mpz = dict(engine.get("mpz_config", {}))
        anisotropic = dict(engine.get("anisotropic_config", {}))
        checks = {
            "theta_deg": (
                float(anisotropic.get("crystal_theta_deg", float("nan"))),
                float(configuration.theta_deg),
                1.0e-10,
            ),
            "front_process_zone_length_m": (
                float(front.get("L_pz", float("nan"))),
                float(configuration.process_zone_length_m),
                1.0e-12,
            ),
            "mpz_length_m": (
                float(mpz.get("length_m", float("nan"))),
                float(configuration.process_zone_length_m),
                1.0e-12,
            ),
            "crack_advance_m": (
                float(front.get("da", float("nan"))),
                float(configuration.da_phys_m),
                1.0e-12,
            ),
            "interaction_length_m": (
                float(metadata.get("interaction_ell_m", float("nan"))),
                float(configuration.interaction_length_m),
                1.0e-12,
            ),
        }
        for name, (actual, expected, tolerance) in checks.items():
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=1.0e-10, abs_tol=tolerance
            ):
                raise ValueError(
                    f"{row['state_id']} mechanics mismatch for {name}: "
                    f"{actual} != {expected}"
                )
        if int(mpz.get("n_bins", -1)) != int(configuration.process_zone_bins):
            raise ValueError(
                f"{row['state_id']} MPZ bins mismatch: "
                f"{mpz.get('n_bins')} != {configuration.process_zone_bins}"
            )
        if len(metadata.get("active_x_m", [])) != int(
            configuration.process_zone_bins
        ):
            raise ValueError(
                f"{row['state_id']} active station count does not match "
                f"process_zone_bins={configuration.process_zone_bins}"
            )
        if metadata.get("fixed_crack_geometry") is not True:
            raise ValueError(f"{row['state_id']} is not a fixed-crack FEM snapshot")
        if metadata.get("active_kernel_supported") is not True:
            raise ValueError(f"{row['state_id']} does not support the active kernel")
        if metadata.get("wake_kernel_supported") is not False:
            raise ValueError(f"{row['state_id']} unexpectedly supports wake shielding")
        required_provenance = {
            "trajectory_state_cloned": True,
            "production_state_mutated": False,
            "plasticity_frozen": True,
            "kinetics_not_advanced": True,
            "hazard_clocks_not_advanced": True,
            "moving_process_zone_not_advanced": True,
            "fractional_moving_frame_not_called": True,
            "endpoint_mesh_reconstructed": True,
            "endpoint_mesh_re_equilibrated": True,
            "production_engine_state_bitwise_unchanged": True,
            "production_fractional_moving_frame_preserved": True,
            "production_mobile_kinetic_solver_preserved": True,
            "measurement_reconstruction_called_engine_step": False,
            "measurement_reconstruction_called_mpz_evolve": False,
            "measurement_reconstruction_called_mpz_advance": False,
            "source_damage_field_interpolated": False,
            "endpoint_caps_excluded": True,
        }
        for key, expected in required_provenance.items():
            if metadata.get(key) is not expected:
                raise ValueError(
                    f"{row['state_id']} capture provenance mismatch for {key}: "
                    f"{metadata.get(key)!r} != {expected!r}"
                )
        if metadata.get("ahead_of_tip_killed_elements") != 0:
            raise ValueError(
                f"{row['state_id']} has reconstructed damage ahead of the accepted tip"
            )
        if metadata.get("kill_radius_floor_m") != 0.0:
            raise ValueError(
                f"{row['state_id']} uses a nonzero crack reconstruction width floor"
            )
        measurement_h = float(
            metadata.get("measurement_mesh_hbar_tip_m", float("nan"))
        )
        trajectory_h = float(
            metadata.get("trajectory_mesh_hbar_tip_m", float("nan"))
        )
        if not math.isfinite(measurement_h) or measurement_h <= 0.0:
            raise ValueError(f"{row['state_id']} lacks a finite measurement hbar_tip")
        if not math.isfinite(trajectory_h) or trajectory_h <= 0.0:
            raise ValueError(f"{row['state_id']} lacks a finite trajectory hbar_tip")
        if configuration.temperature_dependent_mechanics:
            actual_temperature = float(metadata.get("temperature_K", float("nan")))
            if not math.isclose(
                actual_temperature,
                float(configuration.temperature_K),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    f"{row['state_id']} temperature mismatch: "
                    f"{actual_temperature} != {configuration.temperature_K}"
                )


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.stdout:
        print(completed.stdout, end="", file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"current-configuration kernel build failed ({completed.returncode}): "
            + " ".join(command)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--snapshot-archive", type=Path)
    parser.add_argument("--load-invariance-root", type=Path)
    parser.add_argument("--load-invariance-archive", type=Path)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--family-out", type=Path, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--da-phys-um", type=float, required=True)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--margin-events", type=float, default=1.0)
    parser.add_argument("--allow-unconverged-capture", action="store_true")
    args = parser.parse_args()

    configuration = load_configuration(args.mechanical_config)
    if (
        configuration.branching_mode != "single_front"
        or configuration.maximum_fronts != 1
    ):
        raise SystemExit("current built-in kernel provider is single-front only")

    outroot = args.outroot.expanduser().resolve()
    family_out = args.family_out.expanduser().resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="v10227_current_kernel_") as temp:
        workspace = Path(temp)
        snapshot_root = BASE._materialize(
            root=args.snapshot_root,
            archive=args.snapshot_archive,
            workspace=workspace,
            kind="snapshots",
        )
        load_root = BASE._materialize(
            root=args.load_invariance_root,
            archive=args.load_invariance_archive,
            workspace=workspace,
            kind="load_invariance",
        )
        self_consistency = _validate_self_consistency_selection(
            snapshot_root,
            configuration,
            allow_unconverged_capture=bool(args.allow_unconverged_capture),
        )
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_v10_2_27_capture_physics_contract.py"),
                "--snapshot-root",
                str(snapshot_root),
                "--mechanical-config",
                str(args.mechanical_config.expanduser().resolve()),
            ]
        )
        states = BASE._state_records(snapshot_root, load_root)
        _validate_artifacts(snapshot_root, states, configuration)

        command = [
            sys.executable,
            str(BASE_PATH),
            "--snapshot-root",
            str(snapshot_root),
            "--load-invariance-root",
            str(load_root),
            "--mechanical-config",
            str(args.mechanical_config.expanduser().resolve()),
            "--outroot",
            str(outroot),
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
        _run(command)

    payload = json.loads(family_out.read_text())
    payload.update(
        {
            "mechanical_configuration": configuration.canonical_payload(),
            "mechanical_configuration_fingerprint": configuration.fingerprint(),
            "kernel_provider_id": configuration.kernel_provider_id,
            "kernel_recalculated_from_current_configuration": True,
            "historical_reference_condition_required": False,
            "production_moving_process_zone_physics_preserved": True,
            "capture_endpoint_reconstruction_is_measurement_only": True,
        }
    )
    family_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    audit = validate_family(
        family_out,
        expected_configuration_fingerprint=configuration.fingerprint(),
    )
    if self_consistency.get("validated") is True:
        expected_physics = self_consistency["selection"][
            "converged_candidate_family_physics_fingerprint"
        ]
        if audit["physics_fingerprint"] != expected_physics:
            raise ValueError(
                "canonical family does not match the converged target-family physics: "
                f"{audit['physics_fingerprint']} != {expected_physics}"
            )

    manifest_path = outroot / "kernel_build_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update(
        {
            "schema": "v10.2.27_current_configuration_kernel_build_v5",
            "configuration": configuration.canonical_payload(),
            "configuration_fingerprint": configuration.fingerprint(),
            "family": audit["family"],
            "family_sha256": audit["file_sha256"],
            "family_physics_fingerprint": audit["physics_fingerprint"],
            "historical_reference_condition_required": False,
            "production_moving_process_zone_physics_preserved": True,
            "capture_endpoint_reconstruction_is_measurement_only": True,
            "self_consistency": self_consistency,
            "production_parameterization_promotion_allowed": bool(
                self_consistency.get("validated", False)
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reused": False, **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
