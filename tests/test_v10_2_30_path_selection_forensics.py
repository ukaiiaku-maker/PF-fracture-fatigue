import json

import numpy as np

from arrhenius_fracture.path_selection_forensics_v10230 import (
    FILENAME, array_hash, candidate_record, record_selection,
)


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
