import hashlib
import json
from pathlib import Path

import pytest

from scripts.qualify_local_incremental_cavity import (
    checkpoint_evidence, qualification, retained_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(tmp_path: Path) -> Path:
    payload = tmp_path / "accepted.state.pkl"
    payload.write_bytes(b"native-accepted-state")
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "schema": "v11.live-fem-branch-checkpoint/1",
        "state_file": payload.name,
        "state_sha256": sha,
        "active_tip_ids": [],
        "sharp_wake_model_id": "sharp_wake_mechanically_separating_v12",
        "topology_transaction_model_id": "native-transaction",
        "event_counters": {"mesh_generation": 5, "topology_actions": 1},
        "equilibrium_observables": {
            "free_dof_residual_relative": 1e-12,
            "energy_identity_residual": 1e-14,
            "total_opening_m": 8e-7,
        },
        "v12_support_state": {"source_commit": "source"},
        "production_void_state": {
            "cavities": [{
                "cavity_id": "void:site-1", "phase": "CONNECTED_VOID",
                "radius_m": 5.5e-5, "connection_entry_m": [0.6e-3, 0.0],
                "connection_exit_m": [0.7e-3, 0.0],
            }],
            "length_ledgers": {
                "fractured_ligament_length_m": 1e-5,
                "connected_void_free_span_m": 1e-4,
                "preexisting_void_free_span_m": 1e-4,
                "active_front_coordinate_advance_m": 1e-5,
            },
        },
    }
    target = tmp_path / "checkpoint.json"
    target.write_text(json.dumps(manifest))
    return target


def test_manifest_first_checkpoint_validation_is_hash_bound(tmp_path):
    target = _checkpoint(tmp_path)
    result = checkpoint_evidence(target)
    assert result["status"] == "NATIVE_CONNECTED_ACCEPTED_STATE_VERIFIED"
    assert result["cavity"]["phase"] == "CONNECTED_VOID"
    payload = target.with_name("accepted.state.pkl")
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        checkpoint_evidence(target)


def test_retained_native_cavity_evidence_is_discriminating():
    result = retained_evidence()
    assert result["complete_ligament"]["drive_source"] == (
        "MATCHED_COMPLETE_LIGAMENT_TOPOLOGY_ENERGY"
    )
    assert result["complete_ligament"]["selected_local_incremental_conformance"] is False
    assert result["complete_ligament"]["current_code_role"].startswith("ACTUAL_KINETIC")
    assert result["finite_enabled_suppressed_interval"]["accepted_exposure_s"] > 0.0
    assert result["combined_branch_cavity"]["native_checkpoint_round_trips_exact"] is True


def test_qualification_does_not_claim_unexecuted_child_acceptance(tmp_path):
    result = qualification(_checkpoint(tmp_path))
    assert result["status"] == "COLLECTIVE_LIGAMENT_RESTART_READY_LOCAL_INCREMENTAL_PENDING"
    assert any(
        item.endswith("continue_unified_one_void_from_checkpoint.py")
        for item in result["next_native_command"]
    )
    assert any("No downstream child acceptance" in item for item in result["not_claimed"])
