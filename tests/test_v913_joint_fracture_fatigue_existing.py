import numpy as np
import pandas as pd

from scripts.analyze_v913_joint_fracture_fatigue_existing import (
    continuous_segment_eligible, nondominated_mask, shared_physics_fingerprint,
    small_cca_pls,
)


def test_shared_fingerprint_ignores_id_but_changes_physics():
    base = {"candidate_id": "one", "cleave_G00_eV": 4.0, "emit_G00_eV": 3.0}
    cols = ["cleave_G00_eV", "emit_G00_eV"]
    a = shared_physics_fingerprint(pd.Series(base), cols)
    renamed = shared_physics_fingerprint(pd.Series({**base, "candidate_id": "two"}), cols)
    changed = shared_physics_fingerprint(pd.Series({**base, "emit_G00_eV": 3.1}), cols)
    assert a == renamed
    assert a != changed


def test_continuous_fatigue_segment_does_not_bridge_censor():
    curve = pd.DataFrame({
        "normalized_f": [.94, .98, 1.02, 1.08, 1.14, 1.21],
        "da_dN_m_per_cycle": [1e-15, 2e-15, np.nan, 4e-14, 6e-14, 8e-14],
        "plot_kind": ["resolved", "resolved", "censor", "resolved", "resolved", "resolved"],
    })
    assert not continuous_segment_eligible(curve)
    curve.loc[2, ["da_dN_m_per_cycle", "plot_kind"]] = [3e-15, "resolved"]
    assert continuous_segment_eligible(curve)


def test_partial_also_splits_fatigue_segment():
    curve = pd.DataFrame({
        "normalized_f": [.94, .98, 1.02, 1.08, 1.14, 1.21],
        "da_dN_m_per_cycle": [1e-15, 2e-15, np.nan, 4e-14, 6e-14, 8e-14],
        "plot_kind": ["resolved", "resolved", "partial", "resolved", "resolved", "resolved"],
    })
    assert not continuous_segment_eligible(curve)


def test_pareto_front_directionality():
    values = np.array([[1., 1.], [2., 1.], [1., 2.], [.5, .5]])
    keep = nondominated_mask(values, [True, True])
    assert keep.tolist() == [False, True, True, False]


def test_small_cca_pls_is_finite_and_bounded():
    rng = np.random.default_rng(913)
    x = rng.normal(size=(20, 2))
    y = np.c_[x[:, 0] + .05 * rng.normal(size=20), -x[:, 1] + .05 * rng.normal(size=20)]
    out = small_cca_pls(x, y)
    assert set(out.method) == {"CCA", "PLS_SVD"}
    assert np.isfinite(out.association).all()
    assert (out.association.abs() <= 1 + 1e-12).all()
