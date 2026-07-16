from __future__ import annotations

from pathlib import Path

from arrhenius_fracture.sharp_front_v10_1 import _resolved_wake_shielding


def test_progression_runner_routes_explicit_boolean_wake_flag():
    text = Path("scripts/run_v10_0_2_three_class_progression.sh").read_text()
    assert "wake_args+=(--wake-shielding)" in text
    assert "wake_args+=(--no-wake-shielding)" in text
    assert "wake routing audit failed" in text


def test_boolean_wake_resolution():
    assert _resolved_wake_shielding([]) is True
    assert _resolved_wake_shielding(["--wake-shielding"]) is True
    assert _resolved_wake_shielding(["--no-wake-shielding"]) is False


def test_csv_compatibility_maps_current_mpz_diagnostic_keys():
    text = Path("arrhenius_fracture/sharp_front_v10_1.py").read_text()
    assert 'data["mpz_K_shield_Pa_sqrt_m"]' in text
    assert 'data["mpz_total_K_shield_Pa_sqrt_m"]' in text
    assert 'data["mpz_wake_retained_total"]' in text
    assert 'data["mpz_wake_retained_count"]' in text
