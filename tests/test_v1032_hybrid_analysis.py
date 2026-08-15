from __future__ import annotations

import math
from pathlib import Path

from scripts.analyze_v1032_hybrid_hcf_lcf import _interval_stats, classify, finish


def test_interval_statistics_preserve_subcycle_transition() -> None:
    stats = _interval_stats([12.0, 2.0, 0.5, 0.05])
    assert stats["event_count"] == 4
    assert stats["subcycle_fraction"] == 0.5
    assert stats["fraction_below_10_cycles"] == 0.75
    assert stats["fraction_below_0p1_cycle"] == 0.25
    assert stats["minimum_interval_cycles"] == 0.05


def test_censor_and_partial_are_never_converted_to_rates() -> None:
    base = {"class": "A", "candidate_id": "x", "deltaK_MPa_sqrt_m": 1,
            "normalized_f": 1, "dimensionality": "2D", "integration_mode": "accelerated",
            "da_dN_m_per_cycle": 1e-9, "cycles_to_target": 1e12, "extension_um": 0,
            "event_count": 0}
    censored = finish({**base, "status": "cycle_censor"})
    partial = finish({**base, "status": "partial"})
    assert censored["plot_kind"] == "censor" and math.isnan(censored["da_dN_m_per_cycle"])
    assert partial["plot_kind"] == "partial" and math.isnan(partial["da_dN_m_per_cycle"])


def test_explicit_cycle_limit_is_a_true_cycle_censor() -> None:
    row = {"class": "D", "candidate_id": "x", "deltaK_MPa_sqrt_m": 100,
           "normalized_f": 5, "dimensionality": "1D", "integration_mode": "explicit",
           "da_dN_m_per_cycle": 1e-9, "cycles_to_target": 5000,
           "extension_um": 16, "event_count": 5, "status": "explicit_cycle_limit"}
    result = finish(row)
    assert result["regime_classification"] == "BELOW_FATIGUE_RESOLUTION"
    assert result["plot_kind"] == "censor"
    assert math.isnan(result["da_dN_m_per_cycle"])


def test_near_monotonic_requires_target_and_dense_subcycle_events() -> None:
    row = {"status": "growth_target_reached", "integration_mode": "explicit",
           "cycles_to_target": 0.4, "subcycle_fraction": 0.9,
           "median_interval_cycles": 0.02}
    assert classify(row) == "NEAR_MONOTONIC_EXPLICIT"
    assert classify({**row, "status": "partial"}) == "PARTIAL_OR_NUMERICAL_UNRESOLVED"


def test_sparse_matrix_is_compatible_with_macos_bash_3() -> None:
    script = Path("scripts/run_v10_2_32_hybrid_sparse_matrix.sh").read_text()
    assert "wait -n" not in script
    assert 'wait "${pids[0]}"' in script


def test_explicit_2d_launcher_bounds_checkpoints_and_supports_same_root_restart() -> None:
    script = Path("scripts/run_v10_2_32_explicit_sparse_case.sh").read_text()
    assert "V10230_COMBINED_CHECKPOINT_KEEP_GENERATIONS" in script
    assert '[[ "$RESTART_DIR" == "$OUTROOT" ]]' in script
    assert "hybrid_restart_contract.json" in script
    assert "analyze_v10_2_30_developed_fatigue_growth.py" in script
