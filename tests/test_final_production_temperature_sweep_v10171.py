from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import arrhenius_fracture


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v10_1_7_1_final_temperature_sweep.sh"
ANALYZER = ROOT / "scripts" / "analyze_v10_1_7_1_final_temperature_sweep.py"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_1_7_1.py"


def _analyzer_module():
    spec = spec_from_file_location("v10171_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_v10171():
    assert arrhenius_fracture.__version__ == "10.1.7.3"


def test_production_runner_defaults_to_three_classes_nine_temperatures_and_500um():
    text = RUNNER.read_text()
    assert 'CLASSES=${CLASSES:-"ceramic weakT DBTT"}' in text
    assert 'TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100"}' in text
    assert 'MODES=${MODES:-"full"}' in text
    assert 'TARGET_EXT_UM=${TARGET_EXT_UM:-500}' in text
    assert 'STEPS=${STEPS:-12000}' in text
    assert "arrhenius_fracture.sharp_front_v10_1_7_1" in text


def test_production_runner_preserves_campaign_physics_and_disables_wake():
    text = RUNNER.read_text()
    assert "CAMPAIGN_BACKSTRESS_SCALE=${CAMPAIGN_BACKSTRESS_SCALE:-1.0}" in text
    assert "CAMPAIGN_REFRESH_SCALE=${CAMPAIGN_REFRESH_SCALE:-1.0}" in text
    assert "--no-wake-shielding" in text
    assert "geometry_source_feedback" in text
    assert "forward_spatial_source_field" in text
    assert "developed_state_cumulative_emitted" in text


def test_production_entry_is_label_only():
    text = ENTRY.read_text()
    assert "Constitutive physics is inherited unchanged from v10.1.7" in text
    assert '"constitutive_change_from_v10_1_7": False' in text
    assert '"geometry_source_feedback": False' in text
    assert '"forward_spatial_source_field": False' in text


def test_optional_ablation_is_initiation_referenced():
    module = _analyzer_module()
    rows = [
        {
            "class": "DBTT", "temperature_K": 1100.0, "mode": "full",
            "K_init_MPa_sqrt_m": 41.0,
            "R_rise_late_MPa_sqrt_m": 9.0,
            "R_rise_final_MPa_sqrt_m": 8.0,
            "R_rise_peak_MPa_sqrt_m": 12.0,
        },
        {
            "class": "DBTT", "temperature_K": 1100.0, "mode": "plasticity_off",
            "K_init_MPa_sqrt_m": 40.7,
            "R_rise_late_MPa_sqrt_m": 8.8,
            "R_rise_final_MPa_sqrt_m": 7.9,
            "R_rise_peak_MPa_sqrt_m": 11.7,
        },
    ]
    paired = module._paired_rows(rows)
    assert len(paired) == 1
    assert abs(paired[0]["plastic_initiation_shift_MPa_sqrt_m"] - 0.3) < 1e-12
    assert abs(paired[0]["plastic_R_rise_late_MPa_sqrt_m"] - 0.2) < 1e-12
