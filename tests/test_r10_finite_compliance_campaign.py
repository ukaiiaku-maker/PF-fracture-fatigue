import json

from scripts.r10_finite_compliance_campaign import (
    csv_row_count, developed_slope_rows, execution_timing, hcf_event_window_rows, hcf_rows,
    native_rows, replicate_rate_rows, search_specification, state_effect_rows,
    state_prefix_rows,
)
from scripts import run_r10_inverse_hcf_batch as hcf_batch
from scripts.run_r10_inverse_hcf_batch import resolved_phase_count


def test_campaign_aggregation_preserves_complete_material_identity(tmp_path):
    native = tmp_path / "native_round_0" / "CAND" / "K.json"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps({
        "candidate_id": "CAND", "same_complete_material_sha256": "a" * 64,
        "paired_plasticity_sha256": "b" * 64, "shape": "compact_independent",
        "mode": "cohesive_mixed", "cycle_history": [{"extension_m": 1e-6}],
        "events": [{}], "stop_category": "TARGET_TRUNCATED_CASCADE",
    }))
    rows = native_rows(tmp_path)
    assert rows[0]["same_complete_material_sha256"] == "a" * 64
    assert rows[0]["paired_plasticity_sha256"] == "b" * 64
    assert rows[0]["event_count"] == 1


def test_campaign_hcf_aggregation_requires_native_schema(tmp_path):
    good = tmp_path / "hcf" / "CAND" / "m" / "result.json"
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps({
        "schema": "r10.czm-owned-hcf-trajectory/1", "candidate_id": "CAND",
        "complete_row_sha256": "c" * 64, "accepted_cycles": 1e8,
        "physical_event_count": 0, "disposition": "PHYSICAL_RUNOUT_TO_DECLARED_HORIZON",
    }))
    ignored = tmp_path / "other" / "result.json"
    ignored.parent.mkdir(); ignored.write_text(json.dumps({"schema": "other"}))
    rows = hcf_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "CAND"
    assert rows[0]["accepted_cycles"] == 1e8


def test_campaign_hcf_aggregation_includes_accepted_progress_without_result(tmp_path):
    registry = tmp_path / "projection" / "projected_registry.csv"
    registry.parent.mkdir()
    registry.write_text(
        "complete_row_sha256,candidate_id,selected_shape,selected_mode,predicted_hcf_K_300K\n"
        + f"{'d' * 64},CAND,compact_independent,cohesive_mixed,10\n")
    progress = tmp_path / "hcf" / "CAND" / "m_1p10000" / "progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({
        "candidate_id": "CAND", "complete_row_sha256": "d" * 64,
        "accepted_cycles": 1234., "frequency_Hz": 100., "temperature_K": 300.,
        "R": .1, "events": [{"cycle": 100., "extension_m": 1e-6,
            "wake_transaction": {"active_mobile_postcommit": 12.,
                                 "active_retained_postcommit": 3.,
                                 "wake_slip": -8.}}],
        "observation_path_m": 60e-6, "plastic_feedback": True,
    }))
    checkpoint = progress.parent/"checkpoint"/"high_cycle_live_checkpoint.json"
    checkpoint.parent.mkdir(); checkpoint.write_text("{}")
    rows = hcf_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["evidence_state"] == "ACCEPTED_PROGRESS_WITHOUT_TERMINAL_RESULT"
    assert rows[0]["accepted_seconds"] == 12.34
    assert rows[0]["Kmax_MPa_sqrt_m"] == 11.
    assert rows[0]["DeltaK_MPa_sqrt_m"] == 9.9
    assert rows[0]["physical_event_count"] == 1
    assert rows[0]["restart_checkpoint"].endswith("high_cycle_live_checkpoint.json")
    assert rows[0]["maximum_active_mobile_postcommit"] == 12.
    assert rows[0]["maximum_active_retained_postcommit"] == 3.
    assert rows[0]["maximum_absolute_wake_slip"] == 8.


def test_event_windows_use_latest_accepted_progress_and_keep_window_meanings(tmp_path):
    registry = tmp_path / "projection" / "projected_registry.csv"
    registry.parent.mkdir()
    registry.write_text(
        "complete_row_sha256,candidate_id,selected_shape,selected_mode,predicted_hcf_K_300K\n"
        + f"{'e' * 64},CAND,compact_independent,cohesive_mixed,20\n")
    case = tmp_path / "hcf" / "CAND" / "m_1p00000"
    case.mkdir(parents=True)
    events = [{"cycle": float(index*100), "extension_m": index*1e-6}
              for index in range(1, 101)]
    (case / "progress.json").write_text(json.dumps({
        "candidate_id": "CAND", "complete_row_sha256": "e" * 64,
        "accepted_cycles": 10000., "frequency_Hz": 1000., "temperature_K": 300.,
        "R": .1, "events": events, "observation_path_m": 120e-6,
        "hazard_seed": 7, "plastic_feedback": True,
    }))
    rows = hcf_event_window_rows(tmp_path)
    assert [row["window_purpose"] for row in rows] == [
        "CANONICAL_DEVELOPED_WINDOW", "LATE_STATIONARITY_DIAGNOSTIC",
        "AGGREGATE_STATIONARITY_DIAGNOSTIC"]
    assert all(abs(row["rate_m_per_cycle"]-1e-8) < 1e-20 for row in rows)
    assert rows[0]["evidence_path"].endswith("progress.json")


def test_event_windows_reject_single_cycle_cascade(tmp_path):
    case = tmp_path / "hcf" / "CAND" / "m_1p00000"
    case.mkdir(parents=True)
    events = [{"cycle": index*1e-4, "extension_m": index*1e-6}
              for index in range(1, 61)]
    (case / "progress.json").write_text(json.dumps({
        "candidate_id": "CAND", "complete_row_sha256": "f" * 64,
        "accepted_cycles": .006, "frequency_Hz": 1000., "temperature_K": 900.,
        "R": .1, "events": events, "observation_path_m": 60e-6,
    }))
    assert hcf_event_window_rows(tmp_path) == []


def test_execution_timing_records_real_elapsed_scope(tmp_path):
    class Args:
        root = tmp_path
        start_utc = "2026-10-18T22:30:09Z"
        deadline_utc = "2026-10-19T10:30:09Z"
        end_utc = "2026-10-19T10:31:09Z"
        final = True
    assert execution_timing(Args()) == 0
    timing = json.loads((tmp_path/"campaign_delivery"/"execution_timing.json").read_text())
    assert timing["productive_elapsed_wall_seconds"] == 43260.
    assert timing["declared_allocation_seconds"] == 43200.
    assert timing["allocation_elapsed"] is True
    assert timing["campaign_status"] == "FINAL"


def test_search_specification_declares_mechanics_mapping():
    mechanics = search_specification()["mechanics_mapping"]
    assert mechanics["tip_radius_m"] == 1e-6
    assert mechanics["status"].endswith("NOT_A_FIT_PARAMETER")


def test_replicate_bounds_do_not_claim_a_confidence_interval():
    base = {"candidate_id": "C", "complete_row_sha256": "a"*64,
        "temperature_K": 300., "R": .1, "frequency_Hz": 1000.,
        "plastic_feedback_reference": "EVOLVING_FULL_STATE",
        "DeltaK_MPa_sqrt_m": 10.}
    rows = [{**base, "developed_rate_m_per_cycle": rate, "path": str(index)}
            for index, rate in enumerate((2e-11, 3e-11))]
    bounds = replicate_rate_rows(rows)[0]
    assert bounds["independent_streams"] == 2
    assert bounds["observed_max_over_min"] == 1.5
    assert bounds["interpretation"] == "OBSERVED_STREAM_RANGE; NOT_A_CONFIDENCE_INTERVAL"


def test_csv_row_count_excludes_header(tmp_path):
    path = tmp_path/"rows.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    assert csv_row_count(path) == 2


def test_state_effects_require_matched_full_and_frozen_rates():
    common = {"candidate_id": "C", "complete_row_sha256": "a"*64,
        "temperature_K": 300., "R": .1, "frequency_Hz": 1000.}
    rows = []
    for feedback, factor in (("EVOLVING_FULL_STATE", 2.),
                             ("FROZEN_INITIAL_PLASTIC_STATE", 1.)):
        for amplitude, rate in ((10., 1e-11*factor), (20., 4e-11*factor)):
            rows.append({**common, "plastic_feedback_reference": feedback,
                "DeltaK_MPa_sqrt_m": amplitude,
                "developed_rate_m_per_cycle": rate})
    effects = state_effect_rows(rows)
    ratios = [row for row in effects if row["comparison"] == "RATE_RATIO"]
    slopes = [row for row in effects if row["comparison"] == "SLOPE_DIFFERENCE"]
    assert len(ratios) == 2 and all(row["C_full_over_frozen"] == 2. for row in ratios)
    assert len(slopes) == 1 and abs(slopes[0]["Delta_m_full_minus_frozen"]) < 1e-15


def test_state_prefix_comparison_reports_only_observed_exposure_bound():
    common = {"candidate_id": "C", "complete_row_sha256": "a"*64,
        "temperature_K": 300., "R": .1, "frequency_Hz": 1000.,
        "DeltaK_MPa_sqrt_m": 10., "hazard_seed": 42}
    rows = [
        {**common, "plastic_feedback_reference": "EVOLVING_FULL_STATE",
         "accepted_cycles": 8000., "first_event_cycle": 4000., "path": "full"},
        {**common, "plastic_feedback_reference": "FROZEN_INITIAL_PLASTIC_STATE",
         "accepted_cycles": 6000., "first_event_cycle": None, "path": "frozen"},
    ]
    comparison = state_prefix_rows(rows)[0]
    assert comparison["comparison"] == "FROZEN_FIRST_EVENT_DELAY_LOWER_BOUND"
    assert comparison["observed_delay_lower_bound"] == 1.5
    assert comparison["frozen_over_full_first_event_cycle"] is None
    assert comparison["interpretation"].endswith("NOT_A_DEVELOPED_RATE_EFFECT")


def test_campaign_uses_newer_accepted_progress_and_recomputes_window(tmp_path):
    directory = tmp_path / "hcf" / "CAND" / "m"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "schema": "r10.czm-owned-hcf-trajectory/1", "candidate_id": "CAND",
        "complete_row_sha256": "d" * 64, "frequency_Hz": 1000.,
        "accepted_cycles": 10., "physical_event_count": 0,
        "disposition": "RESOURCE_STOP", "observation_path_m": 120e-6,
    }))
    events = [{"cycle": float(index), "extension_m": index*1e-6}
              for index in range(1, 61)]
    (directory / "progress.json").write_text(json.dumps({
        "accepted_cycles": 60., "frequency_Hz": 1000., "events": events,
        "observation_path_m": 120e-6, "unsupported": False,
    }))
    row = hcf_rows(tmp_path)[0]
    assert row["evidence_state"] == "ACCEPTED_PROGRESS_NEWER_THAN_RESULT"
    assert row["physical_event_count"] == 60
    assert row["developed_rate_m_per_cycle"] == 1e-6


def test_developed_slopes_require_two_distinct_amplitudes():
    common = {"candidate_id": "C", "complete_row_sha256": "h",
              "temperature_K": 300., "R": .1, "frequency_Hz": 1000.,
              "plastic_feedback_reference": "EVOLVING_FULL_STATE"}
    rows = [
        {**common, "DeltaK_MPa_sqrt_m": 9., "developed_rate_m_per_cycle": 1e-10},
        {**common, "DeltaK_MPa_sqrt_m": 9., "developed_rate_m_per_cycle": 2e-10},
        {**common, "DeltaK_MPa_sqrt_m": 18., "developed_rate_m_per_cycle": 12e-10},
    ]
    slopes = developed_slope_rows(rows)
    assert len(slopes) == 1
    assert slopes[0]["low_replicates"] == 2
    assert slopes[0]["local_log_slope"] == 3.0


def test_search_spec_keeps_diagnostic_tail_out_of_native_shape_families():
    spec = search_specification()
    assert "compact_independent" in spec["shape_families"]
    assert "algebraic_tail_p_1p1" in spec["diagnostic_only_exclusions"]
    assert spec["strict_internal_reference"]["minimum_connected_rate_decades"] == 2.0
    assert spec["immutable_physics"][0] == "EXP_floor_dislocation_emission"


def test_hcf_phase_resolution_uses_accessible_cycle_fraction():
    row = {"predicted_hcf_accessible_phase_fraction": "0.0067"}
    assert resolved_phase_count(row, 16, 4, 4096) == 598
    assert resolved_phase_count(row, 16, 0, 4096) == 16
    assert resolved_phase_count({}, 16, 4, 4096) == 16


def test_hcf_phase_resolution_recomputes_each_protocol_load(monkeypatch):
    monkeypatch.setattr(hcf_batch, "thermal_from", lambda cohesive: "thermal")
    monkeypatch.setattr(hcf_batch, "mechanics", lambda thermal, temperature: (None, 7.))
    def fake_cycle(Kmax, stiffness, thermal, temperature, shape, area,
                   attempt, **kwargs):
        assert (Kmax, stiffness, thermal, temperature, shape) == (
            12.5, 7., "thermal", 600., "compact_independent")
        return {"accessible_phase_fraction": .125}
    monkeypatch.setattr(hcf_batch, "cyclic_rate", fake_cycle)
    row={"cohesive_parameter_vector":json.dumps({
            "activation_area_m2":1e-19,"attempt_frequency_s":1e12}),
         "selected_shape":"compact_independent"}
    fraction=hcf_batch.accessible_fraction_for_load(row,12.5,600.,.3,100.)
    assert fraction == .125
    assert resolved_phase_count(
        {"predicted_hcf_accessible_phase_fraction":fraction},16,4,256) == 32
