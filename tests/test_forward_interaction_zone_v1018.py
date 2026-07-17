from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

import arrhenius_fracture
from arrhenius_fracture.forward_interaction_zone_tip import (
    _shift_source_field_with_virgin_inflow,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v10_1_8_forward_zone_matrix.sh"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_1_8.py"
MODEL = ROOT / "arrhenius_fracture" / "forward_interaction_zone_tip.py"
ANALYZER = ROOT / "scripts" / "analyze_v10_1_8_forward_zone.py"


def _analyzer_module():
    spec = spec_from_file_location("v1018_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_v1018():
    assert arrhenius_fracture.__version__ == "10.1.8"


def test_uniform_virgin_field_is_invariant_under_fractional_advance():
    field = np.zeros((2, 8))
    field[:, :6] = 2.0
    shifted, outflow, inflow = _shift_source_field_with_virgin_inflow(
        field, distance_m=0.25, dx=1.0, n_interaction_bins=6,
        virgin_count_per_bin=np.array([2.0, 2.0]),
    )
    np.testing.assert_allclose(shifted, field)
    assert np.isclose(outflow, 1.0)
    assert np.isclose(inflow, 1.0)


def test_depleted_field_moves_toward_tip_and_far_edge_is_virgin():
    field = np.zeros((1, 6))
    field[0, :4] = [0.0, 1.0, 2.0, 3.0]
    shifted, outflow, inflow = _shift_source_field_with_virgin_inflow(
        field, distance_m=1.0, dx=1.0, n_interaction_bins=4,
        virgin_count_per_bin=np.array([4.0]),
    )
    np.testing.assert_allclose(shifted[0, :4], [1.0, 2.0, 3.0, 4.0])
    assert np.isclose(outflow, 0.0)
    assert np.isclose(inflow, 4.0)


def test_full_zone_replacement_restores_virgin_field():
    field = np.zeros((2, 5))
    shifted, outflow, inflow = _shift_source_field_with_virgin_inflow(
        field, distance_m=10.0, dx=1.0, n_interaction_bins=4,
        virgin_count_per_bin=np.array([1.5, 2.5]),
    )
    np.testing.assert_allclose(shifted[0, :4], 1.5)
    np.testing.assert_allclose(shifted[1, :4], 2.5)
    np.testing.assert_allclose(shifted[:, 4], 0.0)
    assert np.isclose(outflow, 0.0)
    assert np.isclose(inflow, 16.0)


def test_model_uses_spatial_local_arrhenius_emission_and_far_boundary_inflow():
    text = MODEL.read_text()
    assert "forward_source_available_field" in text
    assert "local_stress_profile_Pa" in text
    assert "manifest.emission.rate" in text
    assert "_shift_source_field_with_virgin_inflow" in text
    assert "temperature_dependent" not in text.lower()


def test_entry_and_runner_keep_wake_out_of_primary_closure():
    entry = ENTRY.read_text()
    runner = RUNNER.read_text()
    assert "temperature_dependent_runtime_source_count" in entry
    assert '"wake_primary_toughening_state": False' in entry
    assert "--no-wake-shielding" in runner
    assert "WAKE_N_BINS=${WAKE_N_BINS:-0}" in runner
    assert 'INTERACTION_SCALES=${INTERACTION_SCALES:-"1 2"}' in runner
    assert 'RETENTION_SCALES=${RETENTION_SCALES:-"1 3"}' in runner
    assert 'TARGET_EXT_UM=${TARGET_EXT_UM:-100}' in runner


def test_analyzer_prefers_high_temperature_forward_development():
    module = _analyzer_module()
    rows = [
        {
            "interaction_length_scale": 1.0,
            "retention_scale": 1.0,
            "temperature_K": 300.0,
            "plastic_R_rise_late_MPa_sqrt_m": 0.1,
            "plastic_initiation_shift_MPa_sqrt_m": 0.2,
            "late_active_mean": 1.0,
            "late_retained_mean": 0.2,
            "cumulative_source_consumed": 4.0,
            "cumulative_source_inflow": 2.0,
            "late_backstress_GPa": 0.1,
            "late_K_shield_MPa_sqrt_m": 0.1,
        },
        {
            "interaction_length_scale": 1.0,
            "retention_scale": 1.0,
            "temperature_K": 1100.0,
            "plastic_R_rise_late_MPa_sqrt_m": 1.8,
            "plastic_initiation_shift_MPa_sqrt_m": 0.3,
            "late_active_mean": 9.0,
            "late_retained_mean": 6.0,
            "cumulative_source_consumed": 20.0,
            "cumulative_source_inflow": 15.0,
            "late_backstress_GPa": 0.5,
            "late_K_shield_MPa_sqrt_m": 0.9,
        },
    ]
    ranked = module._rank(rows, [300.0, 1100.0])
    assert len(ranked) == 1
    assert ranked[0]["candidate_pass"] is True
    assert ranked[0]["late_active_contrast"] == 8.0
    assert ranked[0]["late_retained_contrast"] == 5.8
