import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _record():
    return json.loads((ROOT / "artifacts/v7_piecewise_equilibrated_stress_v2/central_dbtt_v7_readiness.json").read_text())


def test_central_dbtt_v7_is_blocked_only_by_piecewise_boundary_traction():
    record = _record()
    assert record["TRACTION_FREE_BOUNDARY_VERIFICATION"] == "FAIL"
    assert record["RAW_CST_TRACTION"] == "RETAINED_DIAGNOSTIC_FAIL"
    assert record["DBTT_SOURCE_READINESS"] == "BLOCKED_WITH_EXACT_FAILURE"
    assert record["exact_failure"] == ["PIECEWISE_BOUNDARY_TRACTION"]
    assert {key for key, passed in record["predicates"].items() if not passed} == {
        "piecewise_boundary_traction"
    }
    assert record["piecewise_traction_D"] > record["piecewise_traction_E"] > 0.05
    assert record["piecewise_traction_relative_D_to_E"] <= 0.05


def test_arc_diagnosis_localizes_raw_residual_away_from_junction():
    for row in _record()["rows"]:
        diagnosis = row["arc_resolved_cavity_traction_v1"]
        summaries = diagnosis["region_summaries"]
        assert diagnosis["edge_count"] == 512
        assert diagnosis["dominant_partition_by_dimensional_l2"] == "fixed_source_window"
        assert summaries["fixed_source_window"]["fraction_of_complete_l2_squared"] > 0.5
        assert summaries["crack_cavity_junction_neighborhood"]["fraction_of_complete_l2_squared"] < 0.003
        assert diagnosis["largest_edge_localization"]["top_10_fraction_of_complete_l2_squared"] < 0.13


def test_failed_source_readiness_prevents_event_or_downstream_work():
    record = _record()
    assert record["actual_production_event_executed"] is False
    assert record["oracle_generated"] is False
    assert record["oneD_changed"] is False
    assert record["paired_trajectories_run"] is False
    assert record["fatigue_started"] is False
    assert record["next_bounded_step"] == "STOP_NUMERICAL_POINT_SOURCE_QUALIFICATION"
