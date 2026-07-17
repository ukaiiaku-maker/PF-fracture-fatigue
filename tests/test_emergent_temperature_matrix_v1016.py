from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import arrhenius_fracture


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v10_1_6_temperature_matrix.sh"
ANALYZER = ROOT / "scripts" / "analyze_v10_1_6_temperature_matrix.py"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_1_6.py"


def _analyzer_module():
    spec = spec_from_file_location("v1016_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_v1016():
    assert arrhenius_fracture.__version__ == "10.1.6.1"


def test_entry_point_changes_no_constitutive_temperature_law():
    text = ENTRY.read_text()
    assert "sharp_front_v10_1_5" in text
    assert "Temperature dependence enters only" in text
    assert "temperature-independent" in text
    assert "source_capacity" not in text
    assert "shielding_target" not in text


def test_matrix_uses_one_common_scale_pair_for_all_temperatures():
    text = RUNNER.read_text()
    assert 'TEMPS=${TEMPS:-"300 700 1100"}' in text
    assert 'CLASSES=${CLASSES:-"ceramic weakT DBTT"}' in text
    assert 'MODES=${MODES:-"full plasticity_off"}' in text
    assert text.count("CAMPAIGN_BACKSTRESS_SCALE=${CAMPAIGN_BACKSTRESS_SCALE:-1.0}") == 1
    assert text.count("CAMPAIGN_REFRESH_SCALE=${CAMPAIGN_REFRESH_SCALE:-1.0}") == 1
    assert "arrhenius_fracture.sharp_front_v10_1_6" in text
    assert "--material-class \"$CLASS\" --temperatures \"$T_K\"" in text
    assert "per-temperature" in text


def test_matrix_contains_matched_no_plasticity_ablation():
    text = RUNNER.read_text()
    assert "plasticity_off)" in text
    assert "--no-tip-plasticity --no-active-shielding" in text
    assert "active_mobile" in text
    assert "campaign_source_budget_consumed" in text


def test_ablation_metric_is_initiation_referenced_before_subtraction():
    module = _analyzer_module()
    rows = [
        {
            "class": "DBTT",
            "temperature_K": 300.0,
            "mode": "full",
            "K_init_MPa_sqrt_m": 10.0,
            "R_rise_final_MPa_sqrt_m": 3.0,
            "R_rise_late_MPa_sqrt_m": 2.0,
            "R_rise_peak_MPa_sqrt_m": 4.0,
            "max_active_population": 5.0,
            "max_emission_backstress_GPa": 1.0,
            "max_active_K_shield_MPa_sqrt_m": 0.5,
            "r_eff_over_r0_at_initiation": 2.0,
        },
        {
            "class": "DBTT",
            "temperature_K": 300.0,
            "mode": "plasticity_off",
            "K_init_MPa_sqrt_m": 8.0,
            "R_rise_final_MPa_sqrt_m": 2.5,
            "R_rise_late_MPa_sqrt_m": 1.5,
            "R_rise_peak_MPa_sqrt_m": 3.5,
            "max_active_population": 0.0,
            "max_emission_backstress_GPa": 0.0,
            "max_active_K_shield_MPa_sqrt_m": 0.0,
            "r_eff_over_r0_at_initiation": 1.0,
        },
    ]
    paired = module._paired_rows(rows)
    assert len(paired) == 1
    row = paired[0]
    assert row["plastic_initiation_shift_MPa_sqrt_m"] == 2.0
    assert row["plastic_R_rise_final_MPa_sqrt_m"] == 0.5
    assert row["plastic_R_rise_late_MPa_sqrt_m"] == 0.5
    assert row["plastic_R_rise_peak_MPa_sqrt_m"] == 0.5


def test_assessment_requires_dbtt_emergence_and_flat_other_classes():
    module = _analyzer_module()
    paired = [
        {"class": "DBTT", "temperature_K": 300.0, "plastic_R_rise_late_MPa_sqrt_m": 0.1},
        {"class": "DBTT", "temperature_K": 1100.0, "plastic_R_rise_late_MPa_sqrt_m": 2.0},
        {"class": "weakT", "temperature_K": 300.0, "plastic_R_rise_late_MPa_sqrt_m": 0.8},
        {"class": "weakT", "temperature_K": 1100.0, "plastic_R_rise_late_MPa_sqrt_m": 1.0},
        {"class": "ceramic", "temperature_K": 300.0, "plastic_R_rise_late_MPa_sqrt_m": 0.0},
        {"class": "ceramic", "temperature_K": 1100.0, "plastic_R_rise_late_MPa_sqrt_m": 0.1},
    ]
    assessment = module._assessment(
        paired,
        dbtt_low_max=0.5,
        dbtt_min_emergence=1.0,
        flat_max_span=1.0,
    )
    assert assessment["DBTT"]["low_T_weak"]
    assert assessment["DBTT"]["high_T_emergent"]
    assert assessment["weakT"]["comparatively_temperature_flat"]
    assert assessment["ceramic"]["comparatively_temperature_flat"]
    assert assessment["overall_pass"]
