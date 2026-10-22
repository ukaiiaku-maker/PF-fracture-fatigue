import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature_data/fatigue_recovered_review/model_conditional/stateful_exp_floor_capacity_policy/native_transfer_joint_search"


def _results():
    return json.loads((OUT / "joint_search.json").read_text())


def test_prediction_table_preserves_all_source_rows_and_chronology():
    with (OUT / "joint_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 760
    for r_value in ("0.1", "0.3", "0.5", "0.7"):
        orders = [int(row["source_order"]) for row in rows if row["R"] == r_value]
        assert orders == sorted(orders)


def test_absolute_and_shape_metrics_recompute_from_packaged_predictions():
    with (OUT / "joint_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = np.log10([float(row["observed_dadn_m_per_cycle"]) for row in rows])
    results = _results()
    for name in ("absolute", "shape"):
        predicted = np.log10([float(row[f"{name}_prediction"]) for row in rows])
        rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        assert np.isclose(rmse, results["finalists"][name]["summary"]["absolute_rmse_log10"], rtol=2e-13)


def test_fixed_kmax_is_a_summary_of_existing_rows_not_extra_data():
    rows = _results()["fixed_Kmax_contrasts"]
    assert len(rows) == 12
    for row in rows:
        expected = row["reported_DeltaK_MPa_sqrt_m"] / (1.0 - row["R"])
        assert np.isclose(row["derived_Kmax_MPa_sqrt_m"], expected, rtol=2e-15)
        assert row["source_series_id"].startswith("lu_fig4_")


def test_native_audit_contains_active_and_failed_strong_escape_envelopes():
    audit = json.loads((OUT / "native_transfer_correspondence.json").read_text())
    labels = {case["label"] for case in audit["cases"]}
    assert {"sustained_active", "strong_escape", "zero_source"} <= labels
    strongest = [case for case in audit["cases"] if case["label"] == "strong_escape" and case["panels"] == 2048][0]
    assert abs(strongest["errors"]["retained_relative"]) > 10.0
    assert audit["moving_frame_probe"]["after"]["wake_retained"] > 0.0


def test_search_keeps_both_pareto_objectives_and_honest_optimizer_status():
    results = _results()
    assert results["search"]["total_profiled_plastic_evaluations"] > 96
    assert not results["search"]["modes"]["absolute"]["success"]
    assert not results["search"]["modes"]["shape"]["success"]
    absolute = results["finalists"]["absolute"]["summary"]
    shape = results["finalists"]["shape"]["summary"]
    assert absolute["absolute_rmse_log10"] < shape["absolute_rmse_log10"]
    assert shape["shape_rmse_log10"] < absolute["shape_rmse_log10"]
