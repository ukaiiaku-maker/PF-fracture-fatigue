import json
import math

from scripts.analyze_r10_12h_scientific_findings import (
    mechanism_comparisons,
    replicate_rows,
    slope_rows,
)


def _row(delta_k, rate, path):
    return {
        "candidate_id": "material",
        "temperature_K": 300.0,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "plastic_feedback_reference": "EVOLVING_FULL_STATE",
        "DeltaK_MPa_sqrt_m": delta_k,
        "developed_rate_m_per_cycle": rate,
        "result_path": path,
    }


def test_slope_rows_collapses_same_amplitude_seed_replicates():
    result = slope_rows([
        _row(10.0, 1.0e-8, "seed-a"),
        _row(10.0, 3.0e-8, "seed-b"),
        _row(20.0, 8.0e-8, "high"),
    ])

    assert len(result) == 1
    assert result[0]["low_replicate_count"] == 2
    assert result[0]["rate_low_m_per_cycle"] == 2.0e-8
    assert result[0]["rate_low_min_m_per_cycle"] == 1.0e-8
    assert result[0]["rate_low_max_m_per_cycle"] == 3.0e-8
    assert math.isclose(result[0]["local_log_slope"], 2.0)


def test_replicate_rows_labels_observed_range_without_confidence_claim():
    result = replicate_rows([
        _row(10.0, 1.0e-8, "seed-a"),
        _row(10.0, 3.0e-8, "seed-b"),
        _row(20.0, 8.0e-8, "single"),
    ])

    assert len(result) == 1
    assert result[0]["independent_streams"] == 2
    assert math.isclose(result[0]["observed_max_over_min"], 3.0)
    assert result[0]["bound_interpretation"] == (
        "OBSERVED_STREAM_RANGE; NOT_A_CONFIDENCE_INTERVAL")


def test_mechanism_comparisons_includes_state_active_11566_control(tmp_path):
    full = (tmp_path / "hcf_round_32_state_active_11566" / "f_1.06" /
            "result.json")
    frozen = (tmp_path / "hcf_mechanism_controls_11566" / "frozen_f_1.06" /
              "result.json")
    for path, cycles in ((full, [0.25]), (frozen, [0.25, 0.5])):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "candidate_id": "R10F_V2_11566__MS",
            "reference_factor": 1.06,
            "accepted_cycles": 1.0e12,
            "events": [{"cycle": cycle} for cycle in cycles],
        }))

    rows = mechanism_comparisons(tmp_path)

    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "R10F_V2_11566__MS"
    assert rows[0]["absolute_difference_cycles"] == 0.0
