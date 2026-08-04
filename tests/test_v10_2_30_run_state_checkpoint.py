from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from arrhenius_fracture.run_state_checkpoint_v10230 import (
    MANIFEST,
    MODEL_ID,
    RestartCheckpointError,
    load_combined_checkpoint,
    validate_compatibility,
    validate_cross_layer,
    write_combined_checkpoint,
)


def fixture(event_count=0, cycles=12.5):
    path = [[0.001, 0.0]]
    for index in range(event_count):
        path.append([0.001 + (index + 1) * 5e-6, 0.0])
    case = {
        "parameter_option": "weakT",
        "temperature_K": 300.0,
        "deltaK_MPa_sqrt_m": 12.0,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "seed": 2001726,
        "da_phys_m": 5e-6,
    }
    outer = {
        "case": case,
        "cycles_total": cycles,
        "geometry": {
            "crack_tip_m": path[-1],
            "front_paths": [path],
            "front_inventory": [{
                "xy": path[-1], "fwd": [1.0, 0.0], "path": path,
                "last_plane": {"t": [0.8, 0.6], "n": [-0.6, 0.8], "name": "p"},
                "win_plane": {"t": [0.8, 0.6], "n": [-0.6, 0.8], "name": "p"},
                "cands": [{"t": [0.8, 0.6], "n": [-0.6, 0.8]}],
            }],
            "committed_event_count": event_count,
            "kinetic_event_index": event_count,
            "transaction_index": event_count,
            "transaction_state": "committed",
            "mesh_metadata": {
                "hbar_m": 1.0e-6,
                "hbar_tip_m": 5.0e-7,
                "tip_reference_centers_m": [[0.001, 0.0]],
            },
        },
    }
    threshold = 2.0
    action = 0.5
    kinetic = {
        "schema": "kinetic-v1",
        "engine_time_s": cycles / 1000.0,
        "geometry_signature": [event_count, event_count * 5e-6, 0.0, event_count * 5e-6],
        "stochastic": {
            "B": action / threshold,
            "hazard_threshold_action": threshold,
            "hazard_action_current": action,
            "hazard_event_index": event_count,
            "hazard_threshold_history": list(range(event_count)),
            "avalanche_base_checkpoint_m": 5e-6,
            "avalanche_event_length_factor": 1.0,
            "avalanche_event_advance_m": 5e-6,
            "rng_state": {"state": 123},
        },
    }
    arrays = {
        "mesh_nodes": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "mesh_elems": np.array([[0, 1, 2]], dtype=np.int64),
        "damage": np.array([1.0, 0.0, 0.0]),
    }
    return outer, arrays, kinetic, np.arange(4.0)


@pytest.mark.parametrize("event_count", [0, 1, 3])
def test_pre_event_and_committed_event_generations_round_trip(tmp_path, event_count):
    outer, arrays, kinetic, vector = fixture(event_count)
    write_combined_checkpoint(tmp_path, outer=outer, arrays=arrays,
                              kinetic=kinetic, kinetic_vector=vector)
    restored_outer, restored_kinetic, restored_arrays = load_combined_checkpoint(tmp_path)
    assert restored_outer["schema"] == MODEL_ID
    assert restored_outer["cycles_total"] == outer["cycles_total"]
    assert restored_outer["geometry"] == outer["geometry"]
    assert restored_kinetic["stochastic"] == kinetic["stochastic"]
    assert np.array_equal(restored_arrays["kinetic_active_vector"], vector)
    assert np.array_equal(restored_arrays["damage"], arrays["damage"])
    validate_cross_layer(restored_outer, restored_kinetic)


def test_uncommitted_transaction_never_replaces_committed_manifest(tmp_path):
    outer, arrays, kinetic, vector = fixture(1)
    first = write_combined_checkpoint(tmp_path, outer=outer, arrays=arrays,
                                      kinetic=kinetic, kinetic_vector=vector)
    pending = tmp_path / "run_state_generations" / ".pending-interrupted"
    pending.mkdir()
    (pending / "outer.json").write_text("partial")
    restored, _, _ = load_combined_checkpoint(tmp_path)
    assert json.loads((tmp_path / MANIFEST).read_text())["generation"] == first["generation"]
    assert restored["geometry"]["transaction_state"] == "committed"


@pytest.mark.parametrize("filename", ["outer.json", "kinetic.json", "state.npz"])
def test_truncated_or_corrupt_generation_fails_closed(tmp_path, filename):
    outer, arrays, kinetic, vector = fixture(1)
    manifest = write_combined_checkpoint(tmp_path, outer=outer, arrays=arrays,
                                         kinetic=kinetic, kinetic_vector=vector)
    target = tmp_path / "run_state_generations" / manifest["generation"] / filename
    target.write_bytes(b"truncated")
    with pytest.raises(RestartCheckpointError, match="corrupt or incomplete"):
        load_combined_checkpoint(tmp_path)


def test_schema_and_case_compatibility_fail_closed(tmp_path):
    outer, arrays, kinetic, vector = fixture()
    write_combined_checkpoint(tmp_path, outer=outer, arrays=arrays,
                              kinetic=kinetic, kinetic_vector=vector)
    manifest_path = tmp_path / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "future"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RestartCheckpointError, match="schema"):
        load_combined_checkpoint(tmp_path)
    for key, wrong in (("parameter_option", "dbtt"), ("seed", 7),
                       ("deltaK_MPa_sqrt_m", 11.0), ("R", 0.2)):
        with pytest.raises(RestartCheckpointError, match="incompatibility"):
            validate_compatibility(outer, {**outer["case"], key: wrong})


def test_cross_layer_event_history_tip_and_clock_mismatches_fail_closed():
    outer, _, kinetic, _ = fixture(2)
    for mutate in (
        lambda o, k: k["stochastic"].__setitem__("hazard_event_index", 1),
        lambda o, k: o["geometry"].__setitem__("crack_tip_m", [0.0, 0.0]),
        lambda o, k: k["stochastic"].__setitem__("B", 0.9),
        lambda o, k: k["stochastic"].__setitem__("hazard_threshold_history", []),
    ):
        changed_outer, changed_kinetic = copy.deepcopy(outer), copy.deepcopy(kinetic)
        mutate(changed_outer, changed_kinetic)
        with pytest.raises(RestartCheckpointError):
            validate_cross_layer(changed_outer, changed_kinetic)


def test_exact_cycles_threshold_and_rng_are_preserved(tmp_path):
    outer, arrays, kinetic, vector = fixture(1, cycles=123456789.125)
    write_combined_checkpoint(tmp_path, outer=outer, arrays=arrays,
                              kinetic=kinetic, kinetic_vector=vector)
    restored_outer, restored_kinetic, _ = load_combined_checkpoint(tmp_path)
    assert restored_outer["cycles_total"] == 123456789.125
    assert restored_kinetic["stochastic"]["hazard_threshold_action"] == 2.0
    assert restored_kinetic["stochastic"]["rng_state"] == {"state": 123}


def test_directional_front_selection_round_trips_without_event2_path_change():
    from arrhenius_fracture.run_state_checkpoint_v10230 import (
        restore_front_state, serialize_front_state,
    )
    engine = object()
    class TransientCycleResult:
        pass
    front = {
        "eng": engine, "xy": np.array([5.1e-4, 1.2e-6]),
        "fwd": np.array([0.98, 0.2]), "path": [np.array([5e-4, 0.0])],
        "last_plane": {"t": np.array([0.8, 0.6]), "n": np.array([-0.6, 0.8])},
        "win_plane": {"t": np.array([0.7, -0.714]), "n": np.array([0.714, 0.7])},
        "cands": [{"t": np.array([0.7, -0.714]), "n": np.array([0.714, 0.7])}],
        "J_source": "cluster", "J_source_code": 0,
        "fatigue_pred_trial": TransientCycleResult(),
    }
    recorded = serialize_front_state(front)
    assert "fatigue_pred_trial" not in recorded
    restored = {"eng": engine}
    restore_front_state(restored, recorded)
    assert restored["eng"] is engine
    assert np.array_equal(restored["last_plane"]["t"], front["last_plane"]["t"])
    assert np.array_equal(restored["win_plane"]["n"], front["win_plane"]["n"])
    assert np.array_equal(restored["cands"][0]["t"], front["cands"][0]["t"])
