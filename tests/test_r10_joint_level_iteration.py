import csv
import json
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.retained_material_r10 import load_offspring_binding
from scripts.design_r10_joint_level_iteration import barrier, effective_rate, moderate_band


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "integration_evidence/r10_joint_search_amendment"
BASE = EVIDENCE / "projection_frontier/strict_projection_frontier.csv"
JOINT = EVIDENCE / "joint_level_iteration/joint_level_candidates.csv"


def test_joint_level_shift_is_exactly_constant_at_300K():
    base = load_offspring_binding(BASE, "R10_DBTT_P40_THERMAL_ANCHOR_V1")
    child = load_offspring_binding(JOINT, "R10J_DBTT_P40_LSHIFT_036MEV_V1")
    stress = np.geomspace(1.0e5, 3.0e10, 101)
    shift = child.manifest.cleavage.values_eV(stress, 300.0) - base.manifest.cleavage.values_eV(stress, 300.0)
    assert shift == pytest.approx(np.full_like(shift, 0.036), abs=2.0e-14)
    assert child.reduced_config.cleavage_hits == base.reduced_config.cleavage_hits
    assert child.reduced_config.cleavage_tau_s == base.reduced_config.cleavage_tau_s


def test_joint_registry_changes_only_declared_opening_fields():
    with BASE.open(newline="") as stream:
        parent_record = next(r for r in csv.DictReader(stream) if r["candidate_id"] == "R10_DBTT_P40_THERMAL_ANCHOR_V1")
    with JOINT.open(newline="") as stream:
        child_record = next(r for r in csv.DictReader(stream) if r["candidate_id"] == "R10J_DBTT_P40_LSHIFT_036MEV_V1")
    parent = json.loads(parent_record["complete_parameter_vector"])
    child = json.loads(child_record["complete_parameter_vector"])
    changed = {key for key in parent if parent[key] != child[key]}
    assert changed == {"candidate_id", "cleave_G00_eV", "cleave_floor_frac"}


def test_moderate_band_requires_contiguous_secants():
    K = np.asarray([5.0, 7.0, 10.0, 15.0, 22.0])
    action = K**4
    (width, low, high), slope_min, slope_max = moderate_band(K, action)
    assert (low, high) == (5.0, 22.0)
    assert width == pytest.approx(np.log10(action[-1] / action[0]))
    assert slope_min == pytest.approx(4.0)
    assert slope_max == pytest.approx(4.0)


def test_gamma_completion_rate_is_finite_and_nonnegative():
    with JOINT.open(newline="") as stream:
        record = next(csv.DictReader(stream))
    surface = barrier(json.loads(record["complete_parameter_vector"]))
    rate = effective_rate(surface, np.asarray([0.0, 1.0e9, 30.0e9]), 300.0, 3.0, 1.0e-6)
    assert np.all(np.isfinite(rate))
    assert np.all(rate >= 0.0)
    assert np.all(np.diff(rate) >= 0.0)
