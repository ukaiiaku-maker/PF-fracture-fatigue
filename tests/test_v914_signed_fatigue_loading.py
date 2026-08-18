from __future__ import annotations

import math

import pytest

from scripts.v914_signed_fatigue_loading import SignedFatigueLoading


def test_positive_R_preserves_historical_waveform_contract() -> None:
    loading = SignedFatigueLoading(
        0.9,
        R=0.1,
        frequency_Hz=1000.0,
        temperature_K=300.0,
        phase_steps=32,
    )
    loading.validate()
    assert math.isclose(loading.K_at_phase(0.0), 0.1, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(loading.K_at_phase(0.5), 1.0, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(loading.K_at_phase(1.0), 0.1, rel_tol=0.0, abs_tol=1e-14)


def test_negative_R_preserves_same_waveform_without_clipping() -> None:
    loading = SignedFatigueLoading(
        1.5,
        R=-0.5,
        frequency_Hz=1000.0,
        temperature_K=300.0,
        phase_steps=32,
    )
    loading.validate()
    assert math.isclose(loading.K_at_phase(0.0), -0.5, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(loading.K_at_phase(0.5), 1.0, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(loading.K_at_phase(1.0), -0.5, rel_tol=0.0, abs_tol=1e-14)


def test_signed_R_bounds_fail_closed() -> None:
    with pytest.raises(ValueError):
        SignedFatigueLoading(
            2.1,
            R=-1.1,
            frequency_Hz=1000.0,
            temperature_K=300.0,
            phase_steps=32,
        ).validate()

    with pytest.raises(ValueError):
        SignedFatigueLoading(
            0.0,
            R=1.0,
            frequency_Hz=1000.0,
            temperature_K=300.0,
            phase_steps=32,
        ).validate()
