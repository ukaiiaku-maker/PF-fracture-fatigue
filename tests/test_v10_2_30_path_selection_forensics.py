import json
import importlib.util
from pathlib import Path

import numpy as np

from arrhenius_fracture.path_selection_forensics_v10230 import (
    FILENAME, array_hash, candidate_record, record_selection,
)
from arrhenius_fracture.mesh import restore_tri_mesh


def candidates():
    return [
        {"name": "cleave", "family": "cleavage", "angle_deg": 10.0,
         "t": np.array([0.984807753012208, 0.17364817766693033]),
         "n": np.array([-0.17364817766693033, 0.984807753012208]),
         "sigma_nn": 3.0, "overdrive": 2.5, "gamma": 1.44},
        {"name": "cleave", "family": "cleavage", "angle_deg": -20.0,
         "t": np.array([0.9396926207859084, -0.3420201433256687]),
         "n": np.array([0.3420201433256687, 0.9396926207859084]),
         "sigma_nn": 2.0, "overdrive": 1.5, "gamma": 1.2},
    ]


def test_candidate_identifier_and_order_are_exactly_repeatable():
    tip = np.array([5e-4, 2e-6])
    first = [candidate_record(row, i, tip, 5e-6) for i, row in enumerate(candidates())]
    second = [candidate_record(row, i, tip, 5e-6) for i, row in enumerate(candidates())]
    assert first == second
    assert [row["stable_candidate_id"] for row in first] == [
        row["stable_candidate_id"] for row in second
    ]
    assert first[0]["projected_endpoint"] == [
        tip[0] + 5e-6 * candidates()[0]["t"][0],
        tip[1] + 5e-6 * candidates()[0]["t"][1],
    ]


def test_forensic_record_is_atomic_and_preserves_selector_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("V10230_PATH_SELECTION_FORENSICS", "1")
    monkeypatch.setenv("EXPECTED_HEAD", "abc123")
    monkeypatch.setenv("CLEAVAGE_HAZARD_SEED", "17")

    class Engine:
        n_adv = 1
        avalanche_event_advance_m = 5e-6
        hazard_threshold_action = 0.4
        hazard_action_current = 0.2

    class Mesh:
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        elems = np.array([[0, 1, 2]])

    front = {"id": 0, "xy": np.array([5e-4, 2e-6]), "fwd": np.array([1.0, 0.0]),
             "path": [np.array([5e-4, 2e-6])], "last_plane": candidates()[0],
             "win_plane": candidates()[0], "eng": Engine()}
    values = np.array([1.0])
    record_selection(outroot=tmp_path, phase="pre_cycle", step=2, cycles=12.0,
                     front=front, sigma2=np.eye(2), all_candidates=candidates(),
                     selected=candidates()[:1], mesh=Mesh(), damage=values,
                     displacement=values, ep_gp=values, rho_gp=values,
                     proposed_length_m=5e-6, threshold=0.4, hazard_action=0.2)
    payload = json.loads((tmp_path / FILENAME).read_text())
    row = payload["records"][0]
    assert row["event_index"] == 2
    assert row["selected_stable_candidate_id"] == row["candidates"][0]["stable_candidate_id"]
    assert row["candidate_order_before_sort"] == row["candidate_order_after_sort"]
    assert row["rng_consumed"] is False
    assert row["hashes"]["damage"] == array_hash(values)


def test_mesh_restore_preserves_selector_radius_anchor():
    points = []
    elems = []
    for index in range(8):
        x = index * 10e-6
        size = (index + 1) * 1e-6
        start = len(points)
        points.extend([[x, 0.0], [x + size, 0.0], [x, size]])
        elems.append([start, start + 1, start + 2])
    nodes = np.asarray(points)
    elems = np.asarray(elems, dtype=np.int32)
    from arrhenius_fracture.mesh import rebuild_tri_mesh
    original = rebuild_tri_mesh(nodes, elems, tip_centers=[[0.0, 0.0]])
    moved_tip_rebuild = rebuild_tri_mesh(nodes, elems, tip_centers=[[70e-6, 0.0]])
    assert original.hbar_tip != moved_tip_rebuild.hbar_tip
    restored = restore_tri_mesh(nodes, elems, {
        "hbar_m": original.hbar,
        "hbar_tip_m": original.hbar_tip,
        "tip_reference_centers_m": [[0.0, 0.0]],
    })
    assert restored.elems.dtype == elems.dtype
    assert restored.hbar_tip == original.hbar_tip


def _comparison_module():
    path = Path(__file__).parents[1] / "scripts" / "compare_v10_2_30_path_selection_forensics.py"
    spec = importlib.util.spec_from_file_location("path_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forensic_comparison_detects_perturbed_selector_input(tmp_path):
    module = _comparison_module()
    control = tmp_path / "control"
    restored = tmp_path / "restored"
    control.mkdir(); restored.mkdir()
    row = {
        "event_index": 2, "execution": "continuous", "combined_checkpoint_generation": "a",
        "selector_input": {"near_tip_stress_tensor_Pa": [[1.0, 0.0], [0.0, 1.0]]},
        "candidates": [], "selected_direction": [1.0, 0.0], "hashes": {"mesh_nodes": "x"},
    }
    payload = {"schema": "v10.2.30_path_selection_forensics_v1", "records": [row]}
    (control / FILENAME).write_text(json.dumps(payload))
    changed = json.loads(json.dumps(payload))
    changed["records"][0]["execution"] = "restored"
    changed["records"][0]["combined_checkpoint_generation"] = "b"
    (restored / FILENAME).write_text(json.dumps(changed))
    assert module.compare(control, restored)["equivalent"] is True
    changed["records"][0]["selector_input"]["near_tip_stress_tensor_Pa"][0][0] = 2.0
    (restored / FILENAME).write_text(json.dumps(changed))
    result = module.compare(control, restored)
    assert result["equivalent"] is False
    assert result["classification"] == "A"
    assert "selector_input" in result["first_difference"]["path"]
