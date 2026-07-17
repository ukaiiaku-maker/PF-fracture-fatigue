from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import arrhenius_fracture
from arrhenius_fracture.geometry_source_feedback_tip import (
    GeometrySourceFeedbackTipEngine,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v10_1_9_1_geometry_source_matrix.sh"
ANALYZER = ROOT / "scripts" / "analyze_v10_1_9_geometry_source.py"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_1_9.py"
MODEL = ROOT / "arrhenius_fracture" / "geometry_source_feedback_tip.py"


def _analyzer_module():
    spec = spec_from_file_location("v10191_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_engine(gain: float, radius: float, r0: float = 1.0):
    engine = object.__new__(GeometrySourceFeedbackTipEngine)
    engine.geometry_source_gain = gain
    engine.geometry_source_base_capacity = np.array([10.0, 10.0])
    engine.geometry_source_reference_radius_m = r0
    engine.geometry_source_first_advance_radius_m = np.nan
    engine.geometry_source_feedback_armed = False
    engine.geometry_source_capacity_ratio = 1.0
    engine.geometry_source_cumulative_exposed = 0.0
    engine.geometry_source_last_exposed = 0.0
    engine.geometry_source_current_normalized_blunting = 0.0
    engine.geometry_source_running_max_normalized_blunting = 0.0
    engine.geometry_source_last_normalized_blunting = 0.0
    engine.mpz = SimpleNamespace(
        site_capacity=np.array([10.0, 10.0]),
        available_sites=np.array([4.0, 5.0]),
        tip_source_activity=np.array([0.4, 0.5]),
        campaign_source_budget_remaining_total=9.0,
        campaign_source_budget_consumed_total=11.0,
    )
    engine.r_eff = lambda: radius
    return engine


def test_version_is_v10191():
    assert arrhenius_fracture.__version__ == "10.1.9.1"


def test_feedback_is_inactive_before_first_advance_arming():
    engine = _minimal_engine(gain=9.0, radius=2.0)
    exposed = engine._apply_geometry_capacity_gain()
    assert exposed == 0.0
    assert engine.geometry_source_current_normalized_blunting == 1.0
    assert engine.geometry_source_running_max_normalized_blunting == 0.0
    np.testing.assert_allclose(engine.mpz.site_capacity, [10.0, 10.0])
    np.testing.assert_allclose(engine.mpz.available_sites, [4.0, 5.0])


def test_first_fired_result_arms_without_exposing_capacity():
    engine = _minimal_engine(gain=9.0, radius=2.0)
    newly_armed = engine._arm_after_first_advance({"fired": True, "r_eff": 2.0})
    assert newly_armed is True
    assert engine.geometry_source_feedback_armed is True
    assert engine.geometry_source_first_advance_radius_m == 2.0
    assert engine.geometry_source_capacity_ratio == 1.0
    assert engine.geometry_source_cumulative_exposed == 0.0
    np.testing.assert_allclose(engine.mpz.site_capacity, [10.0, 10.0])


def test_already_developed_blunting_is_exposed_on_next_interval():
    engine = _minimal_engine(gain=1.0, radius=2.0)
    engine._arm_after_first_advance({"fired": True, "r_eff": 2.0})
    exposed = engine._apply_geometry_capacity_gain()
    # r_eff/r0=2, normalized blunting=1, saturation=1/2, ratio=1.5.
    assert np.isclose(engine.geometry_source_capacity_ratio, 1.5)
    assert np.isclose(engine.geometry_source_running_max_normalized_blunting, 1.0)
    assert np.isclose(exposed, 10.0)


def test_zero_gain_is_exact_capacity_noop_after_arming():
    engine = _minimal_engine(gain=0.0, radius=4.0)
    engine.geometry_source_feedback_armed = True
    exposed = engine._apply_geometry_capacity_gain()
    assert exposed == 0.0
    assert engine.geometry_source_capacity_ratio == 1.0
    np.testing.assert_allclose(engine.mpz.site_capacity, [10.0, 10.0])


def test_bounded_geometry_gain_adds_only_new_capacity():
    engine = _minimal_engine(gain=4.0, radius=1.5)
    engine.geometry_source_feedback_armed = True
    exposed = engine._apply_geometry_capacity_gain()
    # normalized blunting=0.5; saturation=1/3; ratio=1+4/3=7/3.
    assert np.isclose(engine.geometry_source_capacity_ratio, 7.0 / 3.0)
    np.testing.assert_allclose(engine.mpz.site_capacity, [70.0 / 3.0, 70.0 / 3.0])
    np.testing.assert_allclose(engine.mpz.available_sites, [52.0 / 3.0, 55.0 / 3.0])
    assert np.isclose(exposed, 80.0 / 3.0)
    assert np.isclose(engine.geometry_source_cumulative_exposed, exposed)


def test_capacity_gain_uses_irreversible_running_maximum():
    engine = _minimal_engine(gain=4.0, radius=2.0)
    engine.geometry_source_feedback_armed = True
    engine._apply_geometry_capacity_gain()
    capacity = engine.mpz.site_capacity.copy()
    assert engine.geometry_source_running_max_normalized_blunting == 1.0
    engine.r_eff = lambda: 1.1
    engine._apply_geometry_capacity_gain()
    np.testing.assert_allclose(engine.mpz.site_capacity, capacity)
    assert engine.geometry_source_running_max_normalized_blunting == 1.0


def test_runner_is_one_parameter_scale_preserving_matrix():
    text = RUNNER.read_text()
    assert 'TEMPS=${TEMPS:-"300 1100"}' in text
    assert 'GEOMETRY_GAINS=${GEOMETRY_GAINS:-"0 1 4 9"}' in text
    assert "TIP_GEOMETRY_SOURCE_GAIN=\"$gain\"" in text
    assert "CAMPAIGN_BACKSTRESS_SCALE=1" in text
    assert "CAMPAIGN_REFRESH_SCALE=1" in text
    assert "--no-wake-shielding" in text
    assert "geometry_feedback_armed_only_after_first_advance" in text
    assert "geometry_running_maximum_blunting" in text


def test_model_references_r0_and_arms_only_after_fired_result():
    text = MODEL.read_text()
    assert "self.f.r0" in text
    assert 'not bool(result.get("fired", False))' in text
    assert "geometry_source_feedback_armed" in text
    assert "geometry_source_running_max_normalized_blunting" in text


def test_entry_point_uses_one_temperature_independent_gain():
    text = ENTRY.read_text()
    assert "TIP_GEOMETRY_SOURCE_GAIN" in text
    assert 'temperature_dependent_geometry_parameter": False' in text
    assert 'geometry_reference": "original unblunted radius r0"' in text
    assert "GeometrySourceFeedbackTipEngine" in text


def test_ranking_rejects_first_passage_change_even_with_large_high_T_rise():
    module = _analyzer_module()
    rows = [
        {
            "geometry_gain": 0.0,
            "temperature_K": 300.0,
            "K_init_MPa_sqrt_m": 15.0,
            "plastic_R_rise_late_MPa_sqrt_m": 0.1,
            "late_capacity_ratio": 1.0,
            "max_capacity_ratio": 1.0,
            "final_cumulative_exposed": 0.0,
            "late_active_mean": 0.1,
            "late_retained_mean": 0.05,
            "late_backstress_GPa": 0.1,
            "late_K_shield_MPa_sqrt_m": 0.1,
        },
        {
            "geometry_gain": 0.0,
            "temperature_K": 1100.0,
            "K_init_MPa_sqrt_m": 40.0,
            "plastic_R_rise_late_MPa_sqrt_m": 0.0,
            "late_capacity_ratio": 1.0,
            "max_capacity_ratio": 1.0,
            "final_cumulative_exposed": 0.0,
            "late_active_mean": 1.0,
            "late_retained_mean": 0.5,
            "late_backstress_GPa": 0.1,
            "late_K_shield_MPa_sqrt_m": 0.1,
        },
        {
            "geometry_gain": 4.0,
            "temperature_K": 300.0,
            "K_init_MPa_sqrt_m": 15.0,
            "plastic_R_rise_late_MPa_sqrt_m": 0.1,
            "late_capacity_ratio": 1.2,
            "max_capacity_ratio": 1.2,
            "final_cumulative_exposed": 2.0,
            "late_active_mean": 0.2,
            "late_retained_mean": 0.1,
            "late_backstress_GPa": 0.1,
            "late_K_shield_MPa_sqrt_m": 0.1,
        },
        {
            "geometry_gain": 4.0,
            "temperature_K": 1100.0,
            "K_init_MPa_sqrt_m": 42.0,
            "plastic_R_rise_late_MPa_sqrt_m": 3.0,
            "late_capacity_ratio": 3.0,
            "max_capacity_ratio": 3.0,
            "final_cumulative_exposed": 20.0,
            "late_active_mean": 10.0,
            "late_retained_mean": 6.0,
            "late_backstress_GPa": 0.8,
            "late_K_shield_MPa_sqrt_m": 1.0,
        },
    ]
    ranked = module._rank(rows, [300.0, 1100.0], 0.5, 1.0, 1.0, 0.01)
    candidate = next(r for r in ranked if r["geometry_gain"] == 4.0)
    assert candidate["high_feedback_activated"] is True
    assert candidate["high_T_developed_pass"] is True
    assert candidate["first_passage_preserved"] is False
    assert candidate["candidate_pass"] is False


def test_activation_assessment_flags_all_nonzero_high_T_cases_inactive():
    module = _analyzer_module()
    ranked = [
        {"geometry_gain": 0.0, "high_max_capacity_ratio": 1.0},
        {"geometry_gain": 1.0, "high_max_capacity_ratio": 1.0},
        {"geometry_gain": 4.0, "high_max_capacity_ratio": 1.0 + 1.0e-8},
    ]
    assessment = module._activation_assessment(ranked, 1.0e-6)
    assert assessment["inactive_parameter_matrix"] is True
    assert assessment["nonzero_high_temperature_feedback_active"] is False
