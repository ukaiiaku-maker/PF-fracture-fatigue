import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import select_v914_prospective_fatigue_loads as selector
from scripts.select_v914_prospective_fatigue_loads import interpolate_fraction


def test_log_cycle_interpolation_recovers_monotone_target():
    fraction = np.array([0.8, 1.0, 1.2])
    cycles = np.array([100.0, 10.0, 1.0])
    assert abs(interpolate_fraction(fraction, cycles, 10.0) - 1.0) < 1e-12
    assert 1.0 < interpolate_fraction(fraction, cycles, 3.0) < 1.2


def test_production_layout_ignores_root_run_contract(tmp_path, monkeypatch):
    (tmp_path / "run_contract.json").write_text(json.dumps({
        "schema": "v914_endurance_knee_state_screen_contract_v1"
    }))
    case = tmp_path / "candidate"
    case.mkdir()
    points = [
        {"fraction": f, "projected_mean_first_passage_cycles": c}
        for f, c in zip(
            (0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
            (1000.0, 100.0, 10.0, 4.0, 2.0, 0.5),
        )
    ]
    (case / "state_screen.json").write_text(json.dumps({
        "schema": "v914_endurance_knee_state_screen_candidate_v1",
        "candidate_id": "candidate",
        "reference_deltaK_MPa_sqrt_m": 20.0,
        "points": points,
    }))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv", ["select", "--screen-root", str(tmp_path), "--out", str(out)]
    )
    assert selector.main() == 0
    selected = Path(out / "prospective_fatigue_adaptive_loads.csv")
    assert selected.is_file()
    assert set(pd.read_csv(selected).candidate_id) == {"candidate"}
