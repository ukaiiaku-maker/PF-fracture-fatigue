from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import special


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "barrier_morphology", ROOT / "scripts/analyze_v914_barrier_fatigue_morphology.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_exact_exp_floor_barrier_and_production_stress_cap():
    row = pd.Series({
        "Tref_K": 300.0,
        "cleave_G00_eV": 2.0, "cleave_gT_eV_per_K": 0.0,
        "cleave_sigc0_GPa": 3.0, "cleave_sT_GPa_per_K": 0.0,
        "cleave_exp_a": 0.5, "cleave_exp_n": 2.0, "cleave_floor_frac": 0.1,
    })
    dk = np.array([0.0, 10.0, 100.0, 200.0])
    got = MOD.exp_floor_barrier(row, dk, "cleave")
    sigma = MOD.sigma_from_deltaK(dk)
    expected = 0.2 + 1.8 * np.exp(-0.5 * (sigma / 3.0e9) ** 2)
    assert got == pytest.approx(expected)
    assert sigma[-1] == pytest.approx(MOD.SIGMA_CAP_PA)
    assert got[-1] == pytest.approx(got[-2])


def test_multihit_cleavage_rate_matches_production_definition():
    barriers = np.array([0.2, 0.8, 2.0])
    elementary = MOD.NU_C * np.exp(-barriers / (MOD.KB_EV * MOD.T_K))
    expected = np.log10(special.gammainc(MOD.MULTIHIT_M, elementary * MOD.MULTIHIT_TAU_S) / MOD.MULTIHIT_TAU_S)
    assert MOD.log10_multihit_cleavage_rate(barriers) == pytest.approx(expected)


def _curve(statuses):
    rows = []
    for i, finite in enumerate(statuses):
        rows.append({
            "normalized_f": 1.0 + .02 * i, "integration_mode": "explicit",
            "is_finite_rate": finite, "da_dN_m_per_cycle": 1e-8 * (i + 1) if finite else np.nan,
            "log10_da_dN": np.log10(1e-8 * (i + 1)) if finite else np.nan,
        })
    return pd.DataFrame(rows)


def test_isolated_mode_censor_does_not_create_false_reentry():
    segments = MOD._contiguous_finite_segments(_curve([True, True, False, True, True]))
    assert len(segments) == 1
    assert len(segments[0]) == 4


def test_consecutive_censors_preserve_arrest_gap():
    segments = MOD._contiguous_finite_segments(_curve([True, True, False, False, True, True]))
    assert [len(x) for x in segments] == [2, 2]


def test_transition_width_never_reports_zero_from_one_sparse_sample():
    x = np.array([0.9, 1.0, 1.1, 1.2])
    slopes = np.array([10.0, 8.0, 2.0, 1.0])
    width = MOD._transition_width(x, slopes, 1.05, 10.0, 1.0)
    assert np.isnan(width) or width > 0


def test_correlation_targets_exclude_fit_uncertainty_columns():
    n = 12
    d = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)],
        "spatial_validation_class": ["NO_MATCHED_2D"] * n,
        "candidate_plot_class": ["SMOOTH_ARRHENIUS"] * n,
        "cleave_G0_eV": np.linspace(1, 2, n),
        "reference_deltaK_MPa_sqrt_m": np.linspace(10, 20, n),
        "fracture_resistance_300K_MPa_sqrt_m": np.linspace(11, 21, n),
        "S_K_HCF": np.linspace(2, 3, n),
        "S_K_HCF_se": np.linspace(.1, .2, n),
        "arrest_reentry_indicator": np.zeros(n),
        "LCF_upturn_indicator": np.zeros(n),
        "near_monotonic_indicator": np.zeros(n),
        "spatial_bifurcation_indicator": np.zeros(n),
    })
    corr, _ = MOD.correlation_tables(d)
    assert "S_K_HCF" in set(corr.response)
    assert "S_K_HCF_se" not in set(corr.response)


def test_small_n_correlations_are_descriptive_only():
    d = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "spatial_validation_class": ["REDUCED_VALID"] * 3,
        "candidate_plot_class": ["A", "C", "ceramic-like"],
        "cleave_G0_eV": [1.0, 2.0, 3.0],
        "reference_deltaK_MPa_sqrt_m": [10.0, 11.0, 12.0],
        "fracture_resistance_300K_MPa_sqrt_m": [10.5, 11.5, 12.5],
        "S_K_HCF": [1.0, 2.0, 3.0],
        "arrest_reentry_indicator": [0, 0, 0],
        "LCF_upturn_indicator": [0, 1, 1],
        "near_monotonic_indicator": [0, 0, 1],
        "spatial_bifurcation_indicator": [0, 0, 0],
    })
    corr, _ = MOD.correlation_tables(d)
    q = corr[(corr.subset == "REDUCED_VALID") & (corr.predictor == "cleave_G0_eV") & (corr.response == "S_K_HCF")].iloc[0]
    assert q.test_status == "EXPLORATORY_N3_N4"
    assert np.isnan(q.spearman_p)
