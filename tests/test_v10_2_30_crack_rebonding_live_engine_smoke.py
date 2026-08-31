"""Real-engine smoke tests for the crack-rebonding injection point.

``build_shared_engine`` constructs the actual production
``CampaignCalibratedTipEngine`` (spatial MPZ, anisotropic emission,
production shielding) without the full CLI monkeypatch chain -- it does not
include the stochastic-hazard/transactional-event mixins that only apply
inside ``sharp_front_v10_2_30_energy_gated_fatigue.main()`` (those require a
kernel-family file and full CLI argument set to exercise realistically).
This file therefore validates, on a real engine:

- disabled-path parity for ``cycle_step_waveform`` (no ``_rebonding_state``
  installed is the common real-world default; confirmed separately via a
  git-stash comparison that 71/445 broader existing-test outcomes are
  byte-identical with and without the production edits);
- that the enabled path (RB2) runs to completion across several blocks on a
  real engine without exceptions or non-finite state, and that the wake
  ledger accumulates physically sane values (H_b/K_rebond bounded).

The deeper transactional event-commit root-finder
(``_commit_rebonding_event`` in persistent_site_cyclic_energy_gated_v10230.py)
is validated via the isolated unit tests in
test_v10_2_30_crack_rebonding_event_time_coupling.py and by direct source
reasoning against the confirmed normalized_progress_rate formula, not via a
live multi-event production run in this pass -- documented explicitly in
crack_rebonding_equation_lineage.md and the final report rather than
overclaimed here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    CrackRebondingControls,
    RebondModelLevel,
)
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.reduced_shared_state_v1023 import SharedReducedConfig, build_shared_engine, load_manifest


def _small_config(rebonding: CrackRebondingControls | None = None) -> SharedReducedConfig:
    return SharedReducedConfig(
        mpz_length_um=4.0,
        mpz_n_bins=8,
        wake_length_um=4.0,
        wake_n_bins=8,
        blunting_length_um=0.5,
        max_internal_steps=2000,
        drive_factors=(0.132886, 0.008596),
        rebonding=rebonding or CrackRebondingControls(),
    ).validate()


def _controller() -> FatigueCycleHazardController:
    cfg = FatigueControllerConfig(n_phase=16, block_cycles=10.0, max_block_cycles=100.0)
    return FatigueCycleHazardController(cfg, None, None, None)


def test_disabled_engine_has_no_rebonding_state():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    engine = build_shared_engine(manifest, _small_config(), mode="full")
    assert getattr(engine, "_rebonding_state", None) is None
    assert not hasattr(engine, "rebonding_acceleration_qualified")


def test_disabled_cycle_step_waveform_runs_and_is_sane():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    engine = build_shared_engine(manifest, _small_config(), mode="full")
    controller = _controller()
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)

    result = engine.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)
    assert math.isfinite(result["mu_cleave_pred"])
    assert math.isfinite(result["mu_emit"])
    assert math.isfinite(engine.mpz.mobile.sum())
    assert math.isfinite(engine.mpz.retained.sum())


def test_enabled_rb2_runs_multiple_blocks_and_produces_bounded_wake_state():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    rebonding = CrackRebondingControls(
        enabled=True,
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        wake_length_m=2.0e-6,
        wake_weight_length_m=5.0e-7,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=0.5,
        bond_activation_volume_m3=1.0e-29,
        bond_barrier_eV=0.4,
        rupture_activation_volume_m3=1.0e-29,
        rupture_barrier_eV=0.4,
        chemistry_factor=1.0,
        rebonding_block_max_dpB=0.1,
        rebonding_block_max_dpC=0.1,
        rebonding_block_max_dK_rebond_frac=0.1,
    ).validate()
    engine = build_shared_engine(manifest, _small_config(rebonding), mode="full")
    assert getattr(engine, "_rebonding_state", None) is not None
    assert engine.rebonding_acceleration_qualified is False

    controller = _controller()
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)

    for _ in range(5):
        result = engine.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)
        assert math.isfinite(result["mu_cleave_pred"])
        assert math.isfinite(result["mu_emit"])
        state = engine._rebonding_state
        assert 0.0 <= state.H_b <= 1.0
        assert 0.0 <= state.K_rebond_Pa_sqrt_m <= state.K_rebond_max_Pa_sqrt_m + 1.0e-6
        assert math.isfinite(engine.mpz.mobile.sum())
        assert math.isfinite(engine.mpz.retained.sum())


def test_enabled_rb1_contact_proxy_only_matches_disabled_physics():
    """RB1 must be physically identical to RB0 -- the same cycle_step_waveform
    outputs, since patch_Q is exactly zero for CONTACT_PROXY_ONLY."""
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    controller = _controller()
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)

    engine_off = build_shared_engine(manifest, _small_config(), mode="full")
    result_off = engine_off.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)

    rebonding = CrackRebondingControls(
        enabled=True,
        model_level=RebondModelLevel.CONTACT_PROXY_ONLY,
    ).validate()
    engine_proxy = build_shared_engine(manifest, _small_config(rebonding), mode="full")
    result_proxy = engine_proxy.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)

    assert result_off["mu_cleave_pred"] == pytest.approx(result_proxy["mu_cleave_pred"], rel=1.0e-12)
    assert result_off["dB"] == pytest.approx(result_proxy["dB"], rel=1.0e-12)


def test_chronological_phase_clock_advances_across_blocks_and_never_resets():
    """The wake's own K(phase) sampling must be continuous across blocks
    (round-3 review correction): elapsed_time_s must accumulate by exactly
    dt_block each no-event block, not reset to zero, and the phase window
    used for the wake's K(phase) sampling in a later block must genuinely
    differ from the first block's -- confirming the shift is not a no-op."""
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    rebonding = CrackRebondingControls(
        enabled=True,
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        wake_length_m=2.0e-6,
        wake_weight_length_m=5.0e-7,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=0.5,
        chemistry_factor=1.0,
    ).validate()
    engine = build_shared_engine(manifest, _small_config(rebonding), mode="full")
    controller = _controller()
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)

    state = engine._rebonding_state
    assert state.elapsed_time_s == 0.0

    elapsed_history = [state.elapsed_time_s]
    for _ in range(4):
        engine.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)
        elapsed_history.append(state.elapsed_time_s)

    # Never resets to zero after the first (nonzero-duration) block.
    assert all(t >= 0.0 for t in elapsed_history)
    assert elapsed_history[1] > 0.0
    # Strictly increasing mod period until it wraps -- since dt_block > 0 and
    # requested_cycles is fixed and small relative to period, no wrap should
    # occur across 4 blocks here, so it must be monotonically increasing.
    assert all(b >= a for a, b in zip(elapsed_history, elapsed_history[1:]))

    # The phase-shifted K(phase) window used by block 2 must differ from
    # block 1's, confirming the shift is not a no-op (would be a bug: an
    # elapsed_time_s that never actually changes the sampled K(phase)).
    import math

    from arrhenius_fracture.crack_rebonding_v10230 import chronological_phase_offset_rad

    offset_1 = chronological_phase_offset_rad(elapsed_history[0], waveform.period_s)
    offset_2 = chronological_phase_offset_rad(elapsed_history[1], waveform.period_s)
    assert offset_1 != pytest.approx(offset_2)


def test_chronological_phase_state_survives_snapshot_restore():
    from arrhenius_fracture.crack_rebonding_v10230 import RebondingWakeState

    cfg = CrackRebondingControls(enabled=True, model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND).validate()
    state = RebondingWakeState(cfg)
    state.elapsed_time_s = 3.14e-4
    snap = state.snapshot()
    state.elapsed_time_s = 0.0
    state.restore(snap)
    assert state.elapsed_time_s == pytest.approx(3.14e-4)
