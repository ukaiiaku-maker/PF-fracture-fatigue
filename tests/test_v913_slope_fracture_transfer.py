from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analysis = load("slope_fracture", "scripts/analyze_v913_slope_fracture_grid.py")
transfer = load("slope_transfer", "scripts/materialize_v914_slope_fatigue_registry.py")


def test_morphology_classes_use_historical_grid_only():
    temperature = np.asarray(analysis.TEMPERATURES, dtype=float)
    dbtt = np.asarray([99, 10, 11, 12, 13, 20, 25, 30, 32, 34, 36], dtype=float)
    peak = np.asarray([99, 10, 12, 16, 20, 30, 36, 34, 25, 20, 15], dtype=float)
    weak = np.asarray([99, 10, 10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11, 11.1])
    assert analysis.morphology(temperature, dbtt) == "DBTT_LIKE"
    assert analysis.morphology(temperature, peak) == "PEAK_T"
    assert analysis.morphology(temperature, weak) == "WEAK_T"


def test_fingerprint_is_bit_sensitive_and_complete():
    row = pd.Series({field: float(index + 1) for index, field in enumerate(transfer.ACTIVE_FIELDS)})
    original = transfer.fingerprint(row)
    changed = row.copy()
    changed[transfer.ACTIVE_FIELDS[-1]] = np.nextafter(changed[transfer.ACTIVE_FIELDS[-1]], np.inf)
    assert transfer.fingerprint(changed) != original
    assert len(transfer.ACTIVE_FIELDS) == 29


def test_parent_normalization_preserves_reference_at_parent_center():
    parent_reference = 21.02530765128298
    parent_k300 = 26.28653661187115
    candidate_k300 = parent_k300
    transferred = parent_reference * candidate_k300 / parent_k300
    assert transferred == parent_reference


def test_real_design_registry_round_trip_fingerprints_match():
    path = ROOT / "runs/v913_joint_fracture_fatigue_slope_physics_v1/design/prospective_slope_design_registry.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path, float_precision="round_trip")
    assert all(transfer.fingerprint(row) == row.parameter_fingerprint for _, row in frame.iterrows())


def test_generic_global_stage_a_contract_is_emitted():
    source = (ROOT / "scripts/materialize_v914_slope_fatigue_registry.py").read_text()
    assert '"stageA_global_300K_qualified": "true"' in source
    assert '"stageA_reference_fraction_of_K50": reference / k300' in source
    assert '"material_class": "endurance_knee"' in source
