import math

from scripts.analyze_r10_12h_scientific_findings import slope_rows


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
