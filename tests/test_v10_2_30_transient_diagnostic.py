import csv
import json
from pathlib import Path

from arrhenius_fracture.persistent_site_cyclic_coupled_audited_v10229 import (
    _coupled_fields,
)
from scripts.analyze_v10_2_30_weakt_transient_diagnostic import analyze


ROOT = Path(__file__).resolve().parents[1]


def _record(index: int) -> dict:
    cycles = 100.0
    B_pre = 0.01 * index
    return {
        "loading_mode": "cyclic",
        "cycles_requested": 1000.0,
        "cycles_consumed": cycles,
        "cycles_unused": 900.0,
        "B_pre": B_pre,
        "B": B_pre + 0.01,
        "dB_block": 0.01,
        "fired": False,
        "coupled_hazard_lambda_end_per_s": 10.0 * (1.0 + 1.0e-5 * index),
        "coupled_hazard_sigma_end_Pa": 100.0 * (1.0 + 1.0e-5 * index),
        "coupled_hazard_shield_end_Pa_sqrt_m": 5.0 * (1.0 + 1.0e-5 * index),
        "persistent_sigma_back_Pa": 20.0 * (1.0 + 1.0e-5 * index),
        "persistent_tip_radius_m": 1.0e-9 * (1.0 + 1.0e-5 * index),
        "state_mobile_count": 3.0 * (1.0 + 1.0e-5 * index),
        "state_retained_count": 4.0 * (1.0 + 1.0e-5 * index),
        "state_emitted_total": 50.0 + index,
        "state_escaped_total": 10.0 + index,
        "coupled_hazard_accepted_segments": 4,
        "coupled_hazard_rejected_splits": 1,
        "coupled_hazard_trial_integrations": 12,
        "coupled_hazard_work_budget_exhausted": True,
        "coupled_hazard_partial_return": True,
        "coupled_hazard_event_localized": False,
        "coupled_hazard_wall_seconds": 2.0,
        "coupled_hazard_segments": [
            {
                "cycles_proposed": 25.0,
                "cycles_consumed": 25.0,
                "cumulative_cycles": 25.0,
                "maximum_error_ratio": 0.5,
            }
        ],
    }


def test_forward_fields_are_preserved_by_coupled_audit_adapter():
    source = {
        "coupled_hazard_forward_marcher": True,
        "coupled_hazard_recursive_bisection": False,
        "coupled_hazard_work_budget_exhausted": True,
        "coupled_hazard_partial_return": True,
        "coupled_hazard_trial_integrations": 12,
        "coupled_hazard_shield_end_Pa_sqrt_m": 3.0,
    }
    assert _coupled_fields(source) == source


def test_transient_analyzer_reports_candidate_without_propagating(tmp_path):
    records = [_record(index) for index in range(4)]
    (tmp_path / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps({"records": records})
    )
    summary = analyze(
        tmp_path,
        horizon_cycles=1.0e10,
        window=4,
        lambda_span_decades=0.02,
        rate_span_decades=0.02,
        state_relative_tol=1.0e-3,
        shield_relative_tol=1.0e-3,
    )
    assert summary["record_count"] == 4
    assert summary["segment_count"] == 4
    assert summary["cumulative_cycles"] == 400.0
    assert summary["work_budget_partial_returns"] == 4
    assert summary["stationarity"]["stationary_candidate"] is True
    assert summary["stationary_tail_propagation_performed"] is False
    assert summary["safe_to_resume_four_class_campaign"] is False
    assert summary["latest"]["dB_per_cycle"] == 1.0e-4
    assert (tmp_path / "v10_2_30_weakt_transient_summary.json").is_file()
    with (tmp_path / "v10_2_30_weakt_transient_history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4


def test_transient_launcher_is_bounded_and_has_no_feedback_patch():
    text = (
        ROOT / "scripts" / "run_v10_2_30_weakt_0p55_transient_diagnostic.sh"
    ).read_text()
    assert "V10230_FEEDBACK_STATE_BLOCK_CONTROL=1" not in text
    assert "V10230_ACTIVE_STATE_BLOCK_CONTROL=1" not in text
    assert "V10230_FORWARD_MAX_ACCEPTED_SEGMENTS" in text
    assert "V10230_FORWARD_MAX_TRIAL_INTEGRATIONS" in text
    assert "MAX_WALL_SECONDS" in text
    assert "process_sample.txt" in text
    assert "stationary_tail_propagation=off" in text
    assert "analyze_v10_2_30_weakt_transient_diagnostic.py" in text
