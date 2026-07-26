#!/usr/bin/env python3
"""Fail-closed audit of accepted v10.2.27 production kernel capture."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import load_configuration


REQUIRED_TRAJECTORY_FLAGS = (
    "audited_persistent_site_engine_preserved",
    "persistent_site_source_preserved",
    "stochastic_first_passage_preserved",
    "variable_event_length_preserved",
    "moving_process_zone_physics_preserved",
    "fractional_moving_frame_preserved",
    "mobile_kinetic_solver_preserved",
    "active_shielding_preserved",
    "signed_active_shielding_preserved",
    "wake_shielding_remains_disabled",
    "production_parameterization_observed_not_modified",
    "trajectory_seed_family_required_to_break_kernel_build_cycle",
    "trajectory_seed_family_used_only_for_production_state_evolution",
)
REQUIRED_SNAPSHOT_FLAGS = {
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
ALLOWED_PATH_SOURCES = {
    "initial_notch_only",
    "accepted_production_polyline",
    "verified_straight_single_front_tip_displacement",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.snapshot_root.expanduser().resolve()
    configuration = load_configuration(args.mechanical_config)
    manifest_path = root / "kernel_capture_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing accepted-production capture manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = configuration.fingerprint()
    if manifest.get("mechanical_configuration_fingerprint") != expected:
        raise SystemExit("capture mechanical-configuration fingerprint mismatch")
    if manifest.get("mechanical_configuration") != configuration.canonical_payload():
        raise SystemExit("capture mechanical-configuration payload mismatch")

    trajectory = dict(manifest.get("trajectory_driver", {}))
    failed_trajectory = [
        name for name in REQUIRED_TRAJECTORY_FLAGS if trajectory.get(name) is not True
    ]
    if trajectory.get("driver") != "audited_v10_2_27_persistent_site_production_stack":
        failed_trajectory.append("driver")
    if trajectory.get("capture_physics_overrides") != []:
        failed_trajectory.append("capture_physics_overrides")
    if trajectory.get("observed_hazard_modes") != ["exponential"]:
        failed_trajectory.append("observed_hazard_modes")
    if trajectory.get("observed_event_length_modes") != ["threshold_scaled"]:
        failed_trajectory.append("observed_event_length_modes")
    if trajectory.get("observed_tip_kinetics_modes") != ["moving_velocity"]:
        failed_trajectory.append("observed_tip_kinetics_modes")
    seed_sha = str(trajectory.get("trajectory_seed_signed_kernel_family_sha256", ""))
    if len(seed_sha) != 64 or any(
        character not in "0123456789abcdef" for character in seed_sha
    ):
        failed_trajectory.append("trajectory_seed_signed_kernel_family_sha256")
    if not str(trajectory.get("trajectory_seed_signed_kernel_family", "")).strip():
        failed_trajectory.append("trajectory_seed_signed_kernel_family")
    if failed_trajectory:
        raise SystemExit(
            "accepted-production capture physics contract failed: "
            + ",".join(sorted(set(failed_trajectory)))
        )

    rows = []
    for path in sorted(root.glob("*/snapshot.json")):
        payload = json.loads(path.read_text())
        failures = [
            name
            for name, expected_value in REQUIRED_SNAPSHOT_FLAGS.items()
            if payload.get(name) is not expected_value
        ]
        engine = dict(payload.get("engine_config", {}))
        observed = {
            "capture_loading_path": engine.get("capture_loading_path"),
            "persistent_site_engine": engine.get("persistent_site_engine_observed"),
            "persistent_site_source": engine.get("persistent_site_source_observed"),
            "cleavage_hazard_mode": engine.get("cleavage_hazard_mode_observed"),
            "cleavage_event_length_mode": engine.get(
                "cleavage_event_length_mode_observed"
            ),
            "tip_kinetics_mode": engine.get("tip_kinetics_mode_observed"),
            "active_shielding": engine.get("active_shielding_observed"),
            "signed_active_shielding": engine.get(
                "signed_active_shielding_observed"
            ),
            "wake_shielding": engine.get("wake_shielding_observed"),
            "moving_process_zone_advection": engine.get(
                "moving_process_zone_advection_observed"
            ),
        }
        expected_observed = {
            "capture_loading_path": "accepted_v10_2_27_production_state_observer",
            "persistent_site_engine": True,
            "persistent_site_source": True,
            "cleavage_hazard_mode": "exponential",
            "cleavage_event_length_mode": "threshold_scaled",
            "tip_kinetics_mode": "moving_velocity",
            "active_shielding": True,
            "signed_active_shielding": True,
            "wake_shielding": False,
            "moving_process_zone_advection": True,
        }
        failures.extend(
            name
            for name, expected_value in expected_observed.items()
            if observed.get(name) != expected_value
        )
        before = payload.get("production_engine_state_sha256_before")
        after = payload.get("production_engine_state_sha256_after")
        if not before or before != after:
            failures.append("production_engine_state_sha256")

        try:
            extension = float(payload.get("crack_extension_m"))
        except (TypeError, ValueError):
            extension = math.nan
            failures.append("crack_extension_m")
        crack_path = payload.get("crack_path_xy_m", [])
        path_source = payload.get("crack_path_source")
        ahead_killed = payload.get("ahead_of_tip_killed_elements")
        kill_floor = payload.get("kill_radius_floor_m")
        if path_source not in ALLOWED_PATH_SOURCES:
            failures.append("crack_path_source")
        if not math.isfinite(extension) or extension < 0.0:
            failures.append("crack_extension_m")
        elif extension > 1.0e-12:
            if not isinstance(crack_path, list) or len(crack_path) < 2:
                failures.append("resolved_nonzero_crack_path")
            if path_source == "initial_notch_only":
                failures.append("nonzero_initial_notch_only")
        if ahead_killed != 0:
            failures.append("ahead_of_tip_killed_elements")
        if kill_floor != 0.0:
            failures.append("kill_radius_floor_m")
        if payload.get("measurement_damage_source") != (
            "initial_notch_plus_resolved_crack_path"
        ):
            failures.append("measurement_damage_source")

        geometry = {
            "crack_extension_m": extension,
            "crack_path_points": len(crack_path) if isinstance(crack_path, list) else 0,
            "crack_path_source": path_source,
            "straight_single_front_path_synthesized": payload.get(
                "straight_single_front_path_synthesized"
            ),
            "endpoint_caps_excluded": payload.get("endpoint_caps_excluded"),
            "ahead_of_tip_killed_elements": ahead_killed,
            "kill_radius_floor_m": kill_floor,
        }
        rows.append(
            {
                "state_id": payload.get("state_id", path.parent.name),
                "snapshot": str(path),
                "observed": observed,
                "geometry": geometry,
                "failures": sorted(set(failures)),
                "passed": not failures,
            }
        )
    if len(rows) < 2:
        raise SystemExit("capture physics audit requires at least two snapshot states")
    failed_states = [row["state_id"] for row in rows if not row["passed"]]

    result = {
        "schema": "v10.2.27_accepted_production_capture_physics_audit_v3",
        "mechanical_configuration_fingerprint": expected,
        "snapshot_root": str(root),
        "state_count": len(rows),
        "all_states_passed": not failed_states,
        "failed_state_ids": failed_states,
        "trajectory_driver": trajectory,
        "states": rows,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "capture_physics_contract_audit.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed_states:
        raise SystemExit(
            "accepted-production capture snapshot contract failed for "
            + ",".join(failed_states)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
