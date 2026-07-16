from __future__ import annotations

from pathlib import Path


def test_progression_runner_routes_explicit_boolean_wake_flag():
    text = Path("scripts/run_v10_0_2_three_class_progression.sh").read_text()
    assert "--wake-shielding" in text
    assert "--no-wake-shielding" in text
    assert "wake_args+=(--wake-shielding)" in text
    assert "wake_args+=(--no-wake-shielding)" in text


def test_driver_mode_audit_records_wake_mode():
    text = Path("arrhenius_fracture/sharp_front_v10_1.py").read_text()
    assert '"wake_shielding"' in text


def test_step_csv_uses_current_mpz_diagnostic_keys():
    text = Path("arrhenius_fracture/sharp_front.py").read_text()
    assert "mpz_total_K_shield_Pa_sqrt_m" in text
    assert "mpz_active_K_shield_Pa_sqrt_m" in text
    assert "mpz_wake_K_shield_Pa_sqrt_m" in text
    assert "mpz_wake_retained_count" in text
