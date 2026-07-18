import json

import numpy as np
import pytest

from arrhenius_fracture import fatigue_v1
from arrhenius_fracture.fixed_deltaK_v1021 import (
    FixedDeltaKConfig,
    fixed_deltaK_audit_payload,
    install_fixed_deltaK_waveform,
    make_fixed_deltaK_waveform_factory,
    reset_fixed_deltaK_audit,
)
from arrhenius_fracture.sharp_front_v10_2_1 import _normalize_output_semantics


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


def test_context_patches_runtime_import_and_restores_symbol():
    original_module = fatigue_v1.FatigueWaveform
    with install_fixed_deltaK_waveform(24.0):
        assert fatigue_v1.FatigueWaveform is not original_module
        from arrhenius_fracture.fatigue_v1 import FatigueWaveform as DriverWaveform

        assert DriverWaveform is fatigue_v1.FatigueWaveform
        wave = DriverWaveform(
            Kmax=1.0e6, R=0.2, frequency_Hz=500.0, closure_clip=True
        )
        assert np.isclose(wave.DeltaK, 24.0e6)
        assert np.isclose(wave.Kmax, 30.0e6)
    assert fatigue_v1.FatigueWaveform is original_module


def test_invalid_R_is_rejected():
    cfg = FixedDeltaKConfig(18.0).validate()
    with pytest.raises(ValueError):
        cfg.target_Kmax_Pa_sqrt_m(1.0)
    with pytest.raises(ValueError):
        cfg.target_Kmax_Pa_sqrt_m(-0.1)


def test_probe_K_is_not_retained_as_fatigue_toughness(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps([{
        "T": 700.0,
        "Kc_first_MPa_sqrt_m": 0.366,
        "n_advances": 6,
        "mode": "brittle",
    }]))
    (tmp_path / "steps_700K.csv").write_text(
        "step,Uapp_m,KJ_Pa_sqrtm,fatigue_cycles,da_block_m\n"
        "1,2e-7,3.66e5,10,0\n"
    )
    (tmp_path / "toughness_vs_temperature.png").write_bytes(b"not-a-real-png")

    result = _normalize_output_semantics(tmp_path, 24.0, 0.1)
    summary = json.loads((tmp_path / "summary.json").read_text())[0]
    assert summary["Kc_first_MPa_sqrt_m"] is None
    assert summary["KJ_probe_at_first_event_MPa_sqrt_m"] == 0.366
    assert summary["fatigue_DeltaK_MPa_sqrt_m"] == 24.0
    assert np.isclose(summary["fatigue_Kmax_MPa_sqrt_m"], 24.0 / 0.9)
    assert summary["mode"] == "fatigue_propagated"

    header = (tmp_path / "steps_700K.csv").read_text().splitlines()[0]
    assert "KJ_probe_Pa_sqrtm" in header
    assert "fatigue_DeltaK_target_Pa_sqrtm" in header
    assert not (tmp_path / "toughness_vs_temperature.png").exists()
    assert result["summary_Kc_first_suppressed"] is True
