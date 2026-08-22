from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_v914_paris_search import best_fit, fit_segment, level0_decision
from propose_v914_paris_surrogate_batch import gp_predict


CONFIG = json.loads((ROOT / "config/v914_paris_search_target.json").read_text())
ARTIFACT = ROOT / "artifacts/v914_paris_reparameterization"


def point(deltaK: float, *, usable: bool = True) -> dict:
    return {
        "DeltaK_MPa_sqrt_m": deltaK,
        "developed_da_dN_m_per_cycle": 1.0e-12 * deltaK**4,
        "developed_point_usable": usable,
        "censor_type": "none" if usable else "cycle_horizon_reached",
        "event_count": 5 if usable else 0,
    }


def test_exact_power_law_fit_recovers_m_and_C():
    values = [point(value) for value in (10.0, 12.0, 14.0, 16.0, 18.0)]
    fit = fit_segment(values)
    assert fit["m"] == pytest.approx(4.0)
    assert fit["C"] == pytest.approx(1.0e-12)
    assert fit["R2"] == pytest.approx(1.0)


def test_valid_paris_region_requires_four_contiguous_usable_points():
    valid = best_fit([point(value) for value in (10.0, 12.0, 14.0, 16.0, 18.0)], CONFIG)
    assert valid["valid_Paris_region"] is True
    assert valid["n_points"] >= 4

    broken = [point(value) for value in (10.0, 12.0, 14.0, 16.0, 18.0)]
    broken[2] = point(14.0, usable=False)
    rejected = best_fit(broken, CONFIG)
    assert rejected["valid_Paris_region"] is False
    assert rejected["n_points"] == 4


def test_two_points_never_define_a_paris_candidate():
    fit = best_fit([point(10.0), point(15.0)], CONFIG)
    assert fit["valid_Paris_region"] is False
    assert fit["n_points"] == 2


def test_recovered_population_and_initial_design_are_audited():
    import csv

    with (ARTIFACT / "candidate_pool.csv").open(newline="") as stream:
        pool = list(csv.DictReader(stream))
    with (ARTIFACT / "candidate_registry.csv").open(newline="") as stream:
        initial = list(csv.DictReader(stream))
    assert len(pool) == 1024
    assert len(initial) == 32
    assert [row["paris_design_role"] for row in initial[:4]] == ["ABCD_A", "ABCD_B", "ABCD_C", "ABCD_D"]
    assert not any(field.startswith("physics__") for field in initial[0])
    assert len([field for field in initial[0] if field.startswith("u__")]) == 22


def test_surrogate_predictor_returns_finite_uncertainty():
    import numpy as np

    x = np.asarray([[0.0], [0.5], [1.0]])
    y = np.asarray([1.0, 0.0, 1.0])
    mean, std, length = gp_predict(x, y, np.asarray([[0.25], [0.75]]))
    assert np.all(np.isfinite(mean))
    assert np.all(std > 0.0)
    assert length > 0.0


def test_level0_never_rejects_from_partial_load_evidence():
    inert = {"status": "cycle_horizon_reached", "event_count": 0, "completed_physical_cycles": 1.0e8}
    assert level0_decision([inert, inert], 3, 1.0) == (False, "LEVEL0_INCOMPLETE", "level0")
    assert level0_decision([inert, inert, inert], 3, 1.0) == (False, "LEVEL0_EARLY_REJECTION", "none")
