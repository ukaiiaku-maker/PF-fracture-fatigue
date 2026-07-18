import numpy as np
import pytest

from arrhenius_fracture import fatigue_v1, sharp_front
from arrhenius_fracture.fixed_deltaK_v1021 import (
    FixedDeltaKConfig,
    fixed_deltaK_audit_payload,
    install_fixed_deltaK_waveform,
    make_fixed_deltaK_waveform_factory,
    reset_fixed_deltaK_audit,
)


def test_target_kmax_is_derived_from_deltaK_and_R():
    cfg = FixedDeltaKConfig(18.0).validate()
    assert np.isclose(cfg.target_Kmax_Pa_sqrt_m(0.1), 20.0e6)


def test_factory_replaces_incoming_kmax_exactly():
    cfg = FixedDeltaKConfig(18.0).validate()
    reset_fixed_deltaK_audit(cfg)
    factory = make_fixed_deltaK_waveform_factory(fatigue_v1.FatigueWaveform, cfg)
    wave = factory(Kmax=99.0e6, R=0.1, frequency_Hz=1000.0, closure_clip=True)
    assert np.isclose(wave.Kmax, 20.0e6)
    assert np.isclose(wave.DeltaK, 18.0e6)
    audit = fixed_deltaK_audit_payload()
    assert audit["waveforms_created"] == 1
    assert audit["maximum_abs_target_error_Pa_sqrt_m"] <= 1.0e-8
    assert np.isclose(audit["incoming_Kmax_min_Pa_sqrt_m"], 99.0e6)


def test_context_patches_driver_and_restores_symbols():
    original_driver = sharp_front.FatigueWaveform
    original_module = fatigue_v1.FatigueWaveform
    with install_fixed_deltaK_waveform(24.0):
        assert sharp_front.FatigueWaveform is not original_driver
        wave = sharp_front.FatigueWaveform(
            Kmax=1.0e6, R=0.2, frequency_Hz=500.0, closure_clip=True
        )
        assert np.isclose(wave.DeltaK, 24.0e6)
        assert np.isclose(wave.Kmax, 30.0e6)
    assert sharp_front.FatigueWaveform is original_driver
    assert fatigue_v1.FatigueWaveform is original_module


def test_invalid_R_is_rejected():
    cfg = FixedDeltaKConfig(18.0).validate()
    with pytest.raises(ValueError):
        cfg.target_Kmax_Pa_sqrt_m(1.0)
    with pytest.raises(ValueError):
        cfg.target_Kmax_Pa_sqrt_m(-0.1)
