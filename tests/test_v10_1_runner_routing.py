from pathlib import Path


def test_forward_zone_runner_routes_three_causal_modes():
    text = Path("scripts/run_v10_1_forward_zone_ablation_700K.sh").read_text()
    assert 'MODES=${MODES:-"full active_shield_off plasticity_off"}' in text
    assert "--tip-kinetics-mode moving_velocity" in text
    assert "--tip-plasticity --active-shielding" in text
    assert "--tip-plasticity --no-active-shielding" in text
    assert "--no-tip-plasticity --no-active-shielding" in text
    assert "--no-wake-shielding" in text
    assert "v10_1_driver_modes.json" in text


def test_runner_keeps_outer_checkpoint_separate_from_internal_translation():
    text = Path("scripts/run_v10_1_forward_zone_ablation_700K.sh").read_text()
    assert "DA_CHECKPOINT_M" in text
    assert "kinetic-max-translation-substep-m" in text
    assert "5.0e-6" in text
