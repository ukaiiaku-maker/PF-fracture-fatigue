import json

from scripts.r10_finite_compliance_campaign import (
    developed_slope_rows, hcf_rows, native_rows, search_specification,
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
