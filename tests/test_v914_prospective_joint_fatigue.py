import numpy as np
import pandas as pd

from scripts.analyze_v914_prospective_joint_fatigue import (
    classify_status,
    developed_from_events,
    morphology_table,
    preferred_hybrid_rates,
)


def test_status_semantics_do_not_merge_censor_and_partial():
    assert classify_status("maximum_cycles_reached", 10)[0] == "cycle_or_hazard_censor"
    assert classify_status("periodic_state_failed", 10)[0] == "partial_or_numerical_unresolved"
    assert classify_status("growth_target_reached", 100)[0] == "developed_target_reached"


def test_developed_event_rate_uses_only_post_development_events():
    events = [
        {"cycles": 1.0, "cumulative_extension_m": 10e-6},
        {"cycles": 3.0, "cumulative_extension_m": 25e-6},
        {"cycles": 8.0, "cumulative_extension_m": 50e-6},
    ]
    assert np.isclose(developed_from_events(events), 25e-6 / 5.0)


def test_developed_event_rate_accepts_explicit_cycle_schema():
    events = [
        {"cumulative_cycles": 1.0, "cumulative_extension_m": 10e-6},
        {"cumulative_cycles": 4.0, "cumulative_extension_m": 25e-6},
        {"cumulative_cycles": 10.0, "cumulative_extension_m": 55e-6},
    ]
    assert np.isclose(developed_from_events(events), 30e-6 / 6.0)


def test_hybrid_selection_uses_candidate_specific_screened_mode():
    rates = pd.DataFrame(
        [
            {"candidate_id": "a", "seed": 1, "normalized_f": 1.05, "integration_mode": "accelerated"},
            {"candidate_id": "a", "seed": 1, "normalized_f": 1.10, "integration_mode": "accelerated"},
            {"candidate_id": "a", "seed": 1, "normalized_f": 1.10, "integration_mode": "explicit"},
        ]
    )
    loads = pd.DataFrame(
        [
            {"candidate_id": "a", "normalized_f": 1.05, "primary_integration_mode": "ACCELERATED"},
            {"candidate_id": "a", "normalized_f": 1.10, "primary_integration_mode": "BOTH_ACCELERATED_AND_EXPLICIT"},
        ]
    )
    chosen = preferred_hybrid_rates(rates, loads)
    assert chosen.integration_mode.tolist() == ["accelerated", "explicit"]


def test_smooth_sparse_curve_is_not_mislabeled_as_endpoint_knee():
    fractions = [0.5, 0.99, 1.02, 1.06, 1.08, 1.10, 1.12, 1.15, 2.0]
    rates = pd.DataFrame(
        {
            "candidate_id": ["a"] * len(fractions),
            "seed": [1] * len(fractions),
            "normalized_f": fractions,
            "deltaK_MPa_sqrt_m": np.asarray(fractions) * 20.0,
            "integration_mode": ["accelerated"] * 4 + ["explicit"] * 5,
            "target_reached": [False] + [True] * 8,
            "status_class": ["cycle_or_hazard_censor"] + ["developed_target_reached"] * 8,
            "developed_da_dN_m_per_cycle": [np.nan] + list(np.logspace(-12, -3, 8)),
        }
    )
    modes = ["ACCELERATED"] * 4 + ["EXPLICIT"] * 5
    regimes = ["SCREEN_LOWER_ENDPOINT", "VHCF_1E6", "HCF_1E4", "RARE_HCF_LOWER", "HCF_LCF_OVERLAP", "TRANSITION_3_TO_10", "LCF_1_TO_3", "SUBCYCLE_0P1_TO_1", "SCREEN_UPPER_ENDPOINT"]
    loads = pd.DataFrame(
        {
            "candidate_id": ["a"] * len(fractions),
            "normalized_f": fractions,
            "deltaK_MPa_sqrt_m": np.asarray(fractions) * 20.0,
            "primary_integration_mode": modes,
            "selection_regime": regimes,
        }
    )
    row = morphology_table(rates, loads).iloc[0]
    assert not row.localized_knee_detected
    assert np.isnan(row.knee_normalized_f)
    assert row.HCF_LCF_transition_normalized_f == 1.08


def test_morphology_aggregates_multiseed_overlap_without_duplicate_x_gradient():
    fractions = [0.9, 1.0, 1.1, 1.1, 1.1, 1.2, 1.3]
    rates = pd.DataFrame(
        {
            "candidate_id": ["a"] * 7,
            "seed": [1, 1, 1, 2, 3, 1, 1],
            "normalized_f": fractions,
            "deltaK_MPa_sqrt_m": np.asarray(fractions) * 20.0,
            "integration_mode": ["accelerated", "accelerated"] + ["explicit"] * 5,
            "target_reached": [False] + [True] * 6,
            "status_class": ["cycle_or_hazard_censor"] + ["developed_target_reached"] * 6,
            "developed_da_dN_m_per_cycle": [np.nan, 1e-10, 1e-8, 1.1e-8, 0.9e-8, 1e-6, 1e-4],
        }
    )
    loads = pd.DataFrame(
        {
            "candidate_id": ["a"] * 5,
            "normalized_f": [0.9, 1.0, 1.1, 1.2, 1.3],
            "deltaK_MPa_sqrt_m": [18, 20, 22, 24, 26],
            "primary_integration_mode": ["ACCELERATED", "ACCELERATED", "BOTH_ACCELERATED_AND_EXPLICIT", "EXPLICIT", "EXPLICIT"],
            "selection_regime": ["SCREEN_LOWER_ENDPOINT", "HCF_1E4", "HCF_LCF_OVERLAP", "LCF_1_TO_3", "SCREEN_UPPER_ENDPOINT"],
        }
    )
    row = morphology_table(rates, loads).iloc[0]
    assert row.finite_developed_points == 4
    assert row.multiseed_overlap_count == 3
    assert np.isfinite(row.multiseed_overlap_CV)
