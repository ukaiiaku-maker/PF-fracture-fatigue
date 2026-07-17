from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import arrhenius_fracture


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v10_1_7_dbtt_tuning_matrix.sh"
ANALYZER = ROOT / "scripts" / "analyze_v10_1_7_dbtt_tuning.py"
ENTRY = ROOT / "arrhenius_fracture" / "sharp_front_v10_1_7.py"
DIAGNOSTIC = ROOT / "arrhenius_fracture" / "developed_state_diagnostic_tip.py"


def _analyzer_module():
    spec = spec_from_file_location("v1017_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_v1017():
    assert arrhenius_fracture.__version__ == "10.1.7.1"


def test_runner_defaults_to_nine_candidates_and_two_temperatures():
    text = RUNNER.read_text()
    assert 'TEMPS=${TEMPS:-"300 1100"}' in text
    assert 'BACKSTRESS_SCALES=${BACKSTRESS_SCALES:-"0.5 1 2"}' in text
    assert 'REFRESH_SCALES=${REFRESH_SCALES:-"0.1 0.3 1"}' in text
    assert "for BACK in $BACKSTRESS_SCALES" in text
    assert "for REFRESH in $REFRESH_SCALES" in text
    assert "for T_K in $TEMPS" in text


def test_runner_reuses_one_no_plasticity_baseline_per_temperature():
    text = RUNNER.read_text()
    assert "$OUTROOT/baseline/T${T_K}_th${THETA}" in text
    assert "--no-tip-plasticity --no-active-shielding" in text
    assert "$OUTROOT/full/bs${BTAG}_rf${RTAG}/T${T_K}_th${THETA}" in text
    assert "arrhenius_fracture.sharp_front_v10_1_7" in text


def test_entry_point_adds_diagnostics_without_constitutive_change():
    text = ENTRY.read_text()
    assert "constitutive physics unchanged" in text.lower()
    assert "DevelopedStateDiagnosticTipEngine" in text
    assert "_protected.ContinuumSourceKineticTipEngine" in text
    assert '"constitutive_change_from_v10_1_6_1": False' in text


def test_diagnostic_contains_cumulative_and_residence_histories():
    text = DIAGNOSTIC.read_text()
    for key in (
        "developed_state_cumulative_emitted",
        "developed_state_cumulative_refreshed",
        "developed_state_cumulative_trapped",
        "developed_state_cumulative_recovered",
        "developed_state_mobile_residence_count_s",
        "developed_state_retained_residence_count_s",
        "developed_state_retained_fraction",
    ):
        assert key in text


def test_candidate_ranking_enforces_low_temperature_guardrail():
    module = _analyzer_module()
    rows = [
        {
            "backstress_scale": 1.0,
            "refresh_scale": 1.0,
            "temperature_K": 300.0,
            "plastic_R_rise_late_MPa_sqrt_m": 0.1,
            "plastic_initiation_shift_MPa_sqrt_m": 0.4,
            "late_active_mean": 0.1,
            "late_retained_mean": 0.05,
            "cumulative_emitted": 5.0,
            "cumulative_refreshed": 1.0,
            "late_backstress_GPa": 0.1,
            "late_K_shield_MPa_sqrt_m": 0.1,
        },
        {
            "backstress_scale": 1.0,
            "refresh_scale": 1.0,
            "temperature_K": 1100.0,
            "plastic_R_rise_late_MPa_sqrt_m": 2.0,
            "plastic_initiation_shift_MPa_sqrt_m": 0.3,
            "late_active_mean": 8.0,
            "late_retained_mean": 5.0,
            "cumulative_emitted": 30.0,
            "cumulative_refreshed": 20.0,
            "late_backstress_GPa": 0.5,
            "late_K_shield_MPa_sqrt_m": 1.5,
        },
        {
            "backstress_scale": 2.0,
            "refresh_scale": 0.1,
            "temperature_K": 300.0,
            "plastic_R_rise_late_MPa_sqrt_m": 1.2,
            "plastic_initiation_shift_MPa_sqrt_m": 0.4,
            "late_active_mean": 6.0,
            "late_retained_mean": 4.0,
            "cumulative_emitted": 25.0,
            "cumulative_refreshed": 20.0,
            "late_backstress_GPa": 0.7,
            "late_K_shield_MPa_sqrt_m": 1.0,
        },
        {
            "backstress_scale": 2.0,
            "refresh_scale": 0.1,
            "temperature_K": 1100.0,
            "plastic_R_rise_late_MPa_sqrt_m": 3.0,
            "plastic_initiation_shift_MPa_sqrt_m": 0.3,
            "late_active_mean": 10.0,
            "late_retained_mean": 6.0,
            "cumulative_emitted": 40.0,
            "cumulative_refreshed": 30.0,
            "late_backstress_GPa": 0.9,
            "late_K_shield_MPa_sqrt_m": 2.0,
        },
    ]
    ranked = module._rank_candidates(rows, [300.0, 1100.0])
    assert ranked[0]["backstress_scale"] == 1.0
    assert ranked[0]["candidate_pass"] is True
    rejected = next(r for r in ranked if r["backstress_scale"] == 2.0)
    assert rejected["low_T_guardrail_pass"] is False
    assert rejected["candidate_pass"] is False
