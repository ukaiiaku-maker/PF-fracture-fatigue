from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v914_calibrated_transport_campaign import (  # noqa: E402
    _write_csv,
    event_rate_rows,
    fit_groups,
    knee_rows,
    local_slope_rows,
    moving_window_rows,
    summarize_result,
    trend_rows,
    write_plots,
)


def _write_result(root: Path, status: str, delta_k: float) -> Path:
    root.mkdir()
    result = {
        "status": status,
        "completed_physical_cycles": 100,
        "cumulative_extension_m": 100.0e-6,
        "hazard_clock_action": 0.25,
        "events": [
            {
                "cumulative_cycles": 10.0,
                "cumulative_extension_m": 20.0e-6,
                "physical_hazard_action": 1.0,
                "retained_total_m2": 12.0,
            },
            {
                "cumulative_cycles": 90.0,
                "cumulative_extension_m": 100.0e-6,
                "physical_hazard_action": 2.0,
                "retained_total_m2": 14.0,
            },
        ],
        "final_reversibility": {
            "reversible_net_source_slip_count": 3.0,
            "reversible_returned_source_slip_count": 0.5,
            "reversible_physical_return_fraction_of_emitted": 0.1,
        },
        "final_gross_activity": {},
    }
    contract = {
        "configuration_label": "CAL-TRANS-IRREV",
        "return_control": "IRREV",
        "candidate": "test",
        "deltaK_MPa_sqrt_m": delta_k,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "temperature_K": 300.0,
        "seed": 1720,
    }
    (root / "result.json").write_text(json.dumps(result))
    (root / "run_contract.json").write_text(json.dumps(contract))
    return root / "result.json"


def test_summary_uses_developed_span_and_distinguishes_work_budget(tmp_path):
    complete = summarize_result(
        _write_result(tmp_path / "complete", "growth_target_reached", 14.0)
    )
    incomplete = summarize_result(
        _write_result(tmp_path / "incomplete", "work_budget_exhausted", 13.0)
    )

    assert complete["campaign_disposition"] == "completed_growth_target"
    assert complete["developed_da_dN_m_per_cycle"] == pytest.approx(1.0e-6)
    assert complete["developed_da_dt_m_per_s"] == pytest.approx(1.0e-3)
    assert complete["retained_density_sum_m2"] == 14.0
    assert math.isnan(complete["retained_line_content"])
    assert incomplete["campaign_disposition"] == "incomplete_or_rejected"
    assert incomplete["censored"] is False
    assert incomplete["accepted_terminal_result"] is False


def test_fit_and_plot_use_only_completed_developed_points(tmp_path):
    rows = []
    for index, delta_k in enumerate((10.0, 12.0, 14.0)):
        row = summarize_result(
            _write_result(
                tmp_path / f"complete_{index}", "growth_target_reached", delta_k
            )
        )
        row["developed_da_dN_m_per_cycle"] = delta_k**3 * 1.0e-12
        row["developed_da_dt_m_per_s"] = (
            row["developed_da_dN_m_per_cycle"] * row["frequency_Hz"]
        )
        rows.append(row)
    rows.append(summarize_result(
        _write_result(tmp_path / "incomplete", "work_budget_exhausted", 16.0)
    ))

    fits = fit_groups(rows)
    assert len(fits) == 1
    assert fits[0]["point_count"] == 3
    assert fits[0]["paris_m"] == pytest.approx(3.0)
    output = tmp_path / "plot.png"
    write_plots(rows, output)
    assert output.is_file() and output.stat().st_size > 0


def test_restart_result_is_excluded_and_event_diagnostics_are_resolved(tmp_path):
    path = _write_result(tmp_path / "restarted", "growth_target_reached", 14.0)
    payload = json.loads(path.read_text())
    payload["restart_count"] = 1
    path.write_text(json.dumps(payload))

    summary = summarize_result(path)
    assert summary["campaign_disposition"] == "unsupported_restart_result"
    assert summary["fresh_uninterrupted"] is False
    assert fit_groups([summary]) == []

    events = event_rate_rows(path)
    assert len(events) == 2
    assert events[0]["event_da_dN_m_per_cycle"] == pytest.approx(2.0e-6)
    assert events[1]["event_da_dN_m_per_cycle"] == pytest.approx(1.0e-6)
    assert events[1]["tortuosity"] == pytest.approx(1.0)
    windows = moving_window_rows(events, window_m=50.0e-6)
    assert len(windows) == 1
    assert windows[0]["window_da_dN_m_per_cycle"] == pytest.approx(1.0e-6)


def test_local_slopes_and_frequency_trends_use_fresh_completed_rows(tmp_path):
    rows = []
    for index, (delta_k, frequency) in enumerate(((10.0, 100.0), (12.0, 100.0), (10.0, 1000.0))):
        row = summarize_result(_write_result(tmp_path / f"case_{index}", "growth_target_reached", delta_k))
        row["frequency_Hz"] = frequency
        row["developed_da_dN_m_per_cycle"] = delta_k**2 / frequency
        rows.append(row)

    slopes = local_slope_rows(rows)
    assert len(slopes) == 1
    assert slopes[0]["local_paris_m"] == pytest.approx(2.0)
    trends = trend_rows(rows, "frequency_Hz")
    assert len(trends) == 1
    assert trends[0]["rate_ratio_right_over_left"] == pytest.approx(0.1)
    assert trends[0]["frequency_exponent_per_cycle"] == pytest.approx(-1.0)


def test_nonmonotone_knee_is_not_forced_into_paris_fit(tmp_path):
    rows = []
    for index, (delta_k, rate) in enumerate(((15.8, 1.0e-15), (16.0, 1.0e-9), (16.2, 1.0e-10))):
        row = summarize_result(_write_result(tmp_path / f"knee_{index}", "growth_target_reached", delta_k))
        row["developed_da_dN_m_per_cycle"] = rate
        rows.append(row)

    assert fit_groups(rows) == []
    knees = knee_rows(rows)
    assert len(knees) == 1
    assert knees[0]["deltaK_left"] == 15.8
    assert knees[0]["deltaK_right"] == 16.0
    assert knees[0]["classification"] == "sharp_kinetic_knee"


def test_empty_csv_output_clears_stale_analysis(tmp_path):
    path = tmp_path / "stale.csv"
    path.write_text("invalid,old\n1,2\n")
    _write_csv(path, [])
    assert path.is_file()
    assert path.read_text() == ""
