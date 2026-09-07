"""Disabled-mode parity for the crack-rebonding injection point.

Scope note: full end-to-end parity (event times/thresholds/accepted
lengths/RNG state through the stochastic-hazard/transactional-event
mixins) requires the complete CLI chain
(``sharp_front_v10_2_30_energy_gated_fatigue.main``), which needs an
external kernel-family file this session does not construct a fixture for.
This file verifies, at the level directly testable against a real
``build_shared_engine``-constructed engine (no kernel family needed):
exact reproducibility of ``cycle_step_waveform`` outputs, and structural
checkpoint-key-set equality when disabled.
"""
from __future__ import annotations

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls
from arrhenius_fracture.crack_rebonding_v10230 import serialize_rebonding_checkpoint
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.reduced_shared_state_v1023 import SharedReducedConfig, build_shared_engine, load_manifest


def _small_config() -> SharedReducedConfig:
    return SharedReducedConfig(
        mpz_length_um=4.0,
        mpz_n_bins=8,
        wake_length_um=4.0,
        wake_n_bins=8,
        blunting_length_um=0.5,
        max_internal_steps=2000,
        drive_factors=(0.132886, 0.008596),
    ).validate()


def _controller() -> FatigueCycleHazardController:
    cfg = FatigueControllerConfig(n_phase=16, block_cycles=10.0, max_block_cycles=100.0)
    return FatigueCycleHazardController(cfg, None, None, None)


def _numeric_result(engine, controller, waveform) -> dict:
    result = engine.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)
    return {k: v for k, v in result.items() if isinstance(v, (int, float))}


def test_disabled_cycle_step_waveform_is_exactly_reproducible():
    """Two independently constructed disabled engines, driven by identical
    inputs, must produce bit-identical numeric output -- the module being
    importable/present must not perturb anything when the feature is off."""
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)

    engine_a = build_shared_engine(manifest, _small_config(), mode="full")
    result_a = _numeric_result(engine_a, _controller(), waveform)

    engine_b = build_shared_engine(manifest, _small_config(), mode="full")
    result_b = _numeric_result(engine_b, _controller(), waveform)

    assert result_a.keys() == result_b.keys()
    for key in result_a:
        assert result_a[key] == result_b[key], key


def test_checkpoint_key_set_unchanged_when_disabled():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    engine = build_shared_engine(manifest, _small_config(), mode="full")
    assert serialize_rebonding_checkpoint(engine) is None

    # Mirrors persistent_site_high_cycle_checkpoint_v10230.write_checkpoint's
    # construction pattern: the crack_rebonding key is only ever added when
    # serialize_rebonding_checkpoint returns non-None.
    payload = {"schema": "test", "diagnostics": {}}
    rebonding_payload = serialize_rebonding_checkpoint(engine)
    if rebonding_payload is not None:
        payload["crack_rebonding"] = rebonding_payload
    assert "crack_rebonding" not in payload


def test_disabled_engine_never_allocates_rebonding_attributes():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    engine = build_shared_engine(manifest, _small_config(), mode="full")
    controller = _controller()
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)
    for _ in range(3):
        engine.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)
    assert getattr(engine, "_rebonding_state", None) is None
    assert getattr(engine, "_rebonding_block_context", None) is None


@pytest.mark.parametrize(
    "field,zero_value",
    [
        ("chemistry_factor", 0.0),
        ("restored_work_of_separation_J_m2", 0.0),
        ("rebond_K_geometry_factor", 0.0),
        ("bond_attempt_frequency_s", 0.0),
    ],
)
def test_zero_effect_configs_match_disabled_baseline_on_real_engine(field, zero_value):
    from arrhenius_fracture.crack_rebonding_kinetics_v10230 import RebondModelLevel

    manifest = load_manifest(candidate_id="DBTT_A0003837")
    controller = _controller()
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)

    engine_off = build_shared_engine(manifest, _small_config(), mode="full")
    baseline = _numeric_result(engine_off, controller, waveform)

    kwargs = dict(
        enabled=True,
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        bond_activation_volume_m3=1.0e-29,
        bond_barrier_eV=0.3,
        rupture_activation_volume_m3=1.0e-29,
        rupture_barrier_eV=0.3,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=1.0,
        bond_attempt_frequency_s=1.0e13,
        chemistry_factor=1.0,
    )
    kwargs[field] = zero_value
    rebonding = CrackRebondingControls(**kwargs).validate()
    engine_on = build_shared_engine(manifest, SharedReducedConfig(
        mpz_length_um=4.0, mpz_n_bins=8, wake_length_um=4.0, wake_n_bins=8,
        blunting_length_um=0.5, max_internal_steps=2000,
        drive_factors=(0.132886, 0.008596), rebonding=rebonding,
    ).validate(), mode="full")
    zero_effect = _numeric_result(engine_on, _controller(), waveform)

    assert baseline["mu_cleave_pred"] == pytest.approx(zero_effect["mu_cleave_pred"], rel=1.0e-9, abs=1.0e-30)
    assert baseline["dB"] == pytest.approx(zero_effect["dB"], rel=1.0e-9, abs=1.0e-30)
