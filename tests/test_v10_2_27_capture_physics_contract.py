from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_v10_2_27_capture_physics_contract.py"


SNAPSHOT_FLAGS = {
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


def _write_capture(root: Path) -> Path:
    configuration = MechanicalKernelConfiguration()
    config_path = root / "mechanical_configuration.json"
    config_path.write_text(
        json.dumps(configuration.canonical_payload(), indent=2, sort_keys=True) + "\n"
    )
    trajectory = {
        "driver": "audited_v10_2_27_persistent_site_production_stack",
        "capture_physics_overrides": [],
        "observed_hazard_modes": ["exponential"],
        "observed_event_length_modes": ["threshold_scaled"],
        "observed_tip_kinetics_modes": ["moving_velocity"],
        "audited_persistent_site_engine_preserved": True,
        "persistent_site_source_preserved": True,
        "stochastic_first_passage_preserved": True,
        "variable_event_length_preserved": True,
        "moving_process_zone_physics_preserved": True,
        "fractional_moving_frame_preserved": True,
        "mobile_kinetic_solver_preserved": True,
        "active_shielding_preserved": True,
        "signed_active_shielding_preserved": True,
        "wake_shielding_remains_disabled": True,
        "production_parameterization_observed_not_modified": True,
        "trajectory_seed_signed_kernel_family": "/portable/bootstrap/family.json",
        "trajectory_seed_signed_kernel_family_sha256": "c" * 64,
        "trajectory_seed_family_required_to_break_kernel_build_cycle": True,
        "trajectory_seed_family_used_only_for_production_state_evolution": True,
    }
    (root / "kernel_capture_manifest.json").write_text(
        json.dumps(
            {
                "mechanical_configuration": configuration.canonical_payload(),
                "mechanical_configuration_fingerprint": configuration.fingerprint(),
                "trajectory_driver": trajectory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    engine = {
        "capture_loading_path": "accepted_v10_2_27_production_state_observer",
        "persistent_site_engine_observed": True,
        "persistent_site_source_observed": True,
        "cleavage_hazard_mode_observed": "exponential",
        "cleavage_event_length_mode_observed": "threshold_scaled",
        "tip_kinetics_mode_observed": "moving_velocity",
        "active_shielding_observed": True,
        "signed_active_shielding_observed": True,
        "wake_shielding_observed": False,
        "moving_process_zone_advection_observed": True,
    }
    for index in range(2):
        state = root / f"E{index:03d}"
        state.mkdir()
        payload = {
            "state_id": state.name,
            "engine_config": engine,
            "production_engine_state_sha256_before": "a" * 64,
            "production_engine_state_sha256_after": "a" * 64,
            **SNAPSHOT_FLAGS,
        }
        (state / "snapshot.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return config_path


def _run(root: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--snapshot-root",
            str(root),
            "--mechanical-config",
            str(config),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_capture_physics_contract_accepts_observed_production_state(tmp_path: Path):
    config = _write_capture(tmp_path)
    completed = _run(tmp_path, config)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    audit = json.loads((tmp_path / "capture_physics_contract_audit.json").read_text())
    assert audit["all_states_passed"] is True
    assert audit["state_count"] == 2


def test_capture_physics_contract_rejects_mutated_mpz_state(tmp_path: Path):
    config = _write_capture(tmp_path)
    snapshot = tmp_path / "E001" / "snapshot.json"
    payload = json.loads(snapshot.read_text())
    payload["production_engine_state_sha256_after"] = "b" * 64
    snapshot.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    completed = _run(tmp_path, config)
    assert completed.returncode != 0
    assert "production_engine_state_sha256" in (completed.stdout + completed.stderr)
