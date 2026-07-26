#!/usr/bin/env python3
"""Fail-closed audit of accepted-production kernel capture physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import load_configuration


REQUIRED_TRAJECTORY_FLAGS = (
    "stochastic_first_passage_preserved",
    "variable_event_length_preserved",
    "moving_process_zone_physics_preserved",
    "fractional_moving_frame_preserved",
    "mobile_kinetic_solver_preserved",
    "active_shielding_preserved",
    "signed_active_shielding_preserved",
    "production_parameterization_observed_not_modified",
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
    if trajectory.get("capture_physics_overrides") != []:
        failed_trajectory.append("capture_physics_overrides")
    if trajectory.get("observed_hazard_modes") != ["exponential"]:
        failed_trajectory.append("observed_hazard_modes")
    if trajectory.get("observed_event_length_modes") != ["threshold_scaled"]:
        failed_trajectory.append("observed_event_length_modes")
    if trajectory.get("observed_tip_kinetics_modes") != ["moving_velocity"]:
        failed_trajectory.append("observed_tip_kinetics_modes")
    if failed_trajectory:
        raise SystemExit(
            "accepted-production capture physics contract failed: "
            + ",".join(failed_trajectory)
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
            "cleavage_hazard_mode": engine.get("cleavage_hazard_mode_observed"),
            "cleavage_event_length_mode": engine.get(
                "cleavage_event_length_mode_observed"
            ),
            "tip_kinetics_mode": engine.get("tip_kinetics_mode_observed"),
            "active_shielding": engine.get("active_shielding_observed"),
            "signed_active_shielding": engine.get(
                "signed_active_shielding_observed"
            ),
            "moving_process_zone_advection": engine.get(
                "moving_process_zone_advection_observed"
            ),
        }
        expected_observed = {
            "capture_loading_path": "accepted_production_state_observer",
            "cleavage_hazard_mode": "exponential",
            "cleavage_event_length_mode": "threshold_scaled",
            "tip_kinetics_mode": "moving_velocity",
            "active_shielding": True,
            "signed_active_shielding": True,
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
        rows.append({
            "state_id": payload.get("state_id", path.parent.name),
            "snapshot": str(path),
            "observed": observed,
            "failures": sorted(set(failures)),
            "passed": not failures,
        })
    if len(rows) < 2:
        raise SystemExit("capture physics audit requires at least two snapshot states")
    failed_states = [row["state_id"] for row in rows if not row["passed"]]

    result = {
        "schema": "v10.2.27_accepted_production_capture_physics_audit_v1",
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
