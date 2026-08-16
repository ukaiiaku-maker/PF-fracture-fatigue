import numpy as np
import pandas as pd

from scripts.analyze_v913_prospective_fracture_causality import (
    morphology,
    read_root_tables,
    response_summary,
)
from scripts.run_v913_prospective_fracture_causality import _finite_summary


def test_snapshot_summary_preserves_extrema_and_sum():
    out = _finite_summary(np.array([1.0, 2.0, 5.0]), "rate")
    assert out == {
        "rate_min": 1.0,
        "rate_mean": 8.0 / 3.0,
        "rate_max": 5.0,
        "rate_sum": 8.0,
    }


def test_morphology_distinguishes_peak_dbtt_and_weak():
    temperature = np.arange(10.0)
    assert morphology(temperature, np.array([1, 2, 4, 7, 9, 8, 7, 5, 3, 2])) == "PEAK_T"
    assert morphology(temperature, np.linspace(1, 12, 10)) == "DBTT_LIKE"
    assert morphology(temperature, np.linspace(10, 12, 10)) == "WEAK_T"


def test_response_summary_requires_exact_historical_grid_plus_k300():
    temperatures = [300, 700, 800, 900, 950, 1000, 1050, 1100, 1200, 1300, 1400]
    cases = pd.DataFrame(
        {
            "candidate_id": ["p"] * 11,
            "temperature_K": temperatures,
            "status": ["complete"] * 11,
            "K_50um_MPa_sqrt_m": [20, 20, 20, 21, 22, 24, 27, 30, 32, 33, 34],
        }
    )
    registry = pd.DataFrame(
        {
            "prospective_candidate_id": ["p"],
            "design_family": ["DBTT"],
            "design_role": ["FEASIBLE_PRIMARY"],
            "target_code": ["F1"],
            "parameter_fingerprint": ["abc"],
            **{column: [value] for column, value in zip(
                (
                    "achieved__F1_delta_mu",
                    "achieved__F2_activation_window_overlap",
                    "achieved__F3_delta_Theta_sigma_900",
                    "achieved__F4_lowT_plastic_bottleneck",
                ),
                (1.0, 0.2, 0.3, 4.0),
            )},
        }
    )
    summary, points = response_summary(cases, registry)
    assert len(summary) == 1
    assert len(points) == 10
    assert summary.iloc[0].historical_grid_complete
    assert summary.iloc[0].morphology_class == "DBTT_LIKE"


def test_analysis_reads_disjoint_production_batches_without_copying(tmp_path):
    roots = [tmp_path / "primary", tmp_path / "confirmation"]
    for index, root in enumerate(roots):
        root.mkdir()
        pd.DataFrame({"candidate_id": [f"c{index}"], "value": [index]}).to_csv(
            root / "table.csv", index=False
        )
    combined = read_root_tables(roots, "table.csv")
    assert combined.candidate_id.tolist() == ["c0", "c1"]
    assert combined.value.tolist() == [0, 1]
