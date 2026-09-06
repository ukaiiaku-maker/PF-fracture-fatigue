"""PX3.5: regression test for a real bug found during the dwell causal
audit mandated by external review after PX3's screen showed an
unexplained ~6.26x acceleration at hold=2ms.

Root cause: persistent_site_coupled_hazard_v10229.py::_phase_statistics's
hazard_coupled branch computed sig_cleave from a CURSOR-ROTATED K array
(waveform.cycle_schedule_from_elapsed) but duration-weighted every value in
its lambdas/raw/barriers arrays using the UNROTATED dt_values array (from
the plain, phase-0-starting waveform.cycle_schedule) -- pairing each
sig_cleave sample with the WRONG bin's duration whenever the cursor-rotated
and unrotated schedules disagree on bin order. This is invisible at
hold=0 (every bin has the same duration base_period_s/n_phase, so which
duration a given K value gets weighted by does not matter), but at hold>0
the appended dwell bin's much larger duration can get paired with an
unrelated sinusoid K value (or vice versa), corrupting lambda_avg_s --
which the adaptive block-stepping search in integrate_state_coupled_
waveform directly consumes to decide segment lengths, hence event timing.

The decisive proof: with ZERO active rebonding patches (mission's
NO_INITIAL_ACTIVE_WAKE precrack mode -- true before any crack-advance
event), K_rebond is provably 0 regardless of restored_work_of_separation_J_m2,
so a finite- and zero-cohesion config MUST produce bit-identical hazard
statistics at every elapsed-time cursor position. Before the fix, they
matched only at cursor=0 (where the rotation happens to be the identity)
and diverged by up to ~8x at other cursors once hold>0. This is exactly
why PX3's dwell-panel S_h values for hold>0 (+0.3551 censored, +0.7969)
were wrong -- not a real state-mediated acceleration effect.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_coupled_hazard_v10229 import _phase_statistics
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)


def _finite_and_zero_configs():
    registry = json.loads((REPO_ROOT / "artifacts/crack_rebonding_part_x_v1/kinetic_regime_registry.json").read_text())
    cfg_dict = dict(registry["rows"]["COMPETING_REVERSIBLE"]["config"])
    cfg_dict["model_level"] = RebondModelLevel(cfg_dict["model_level"])
    cfg_dict["contact_model"] = ContactModel(cfg_dict["contact_model"])
    cfg_dict["feedback_mode"] = FeedbackMode(cfg_dict["feedback_mode"])
    cfg_dict["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(cfg_dict["initial_precrack_wake_mode"])
    base_cfg = CrackRebondingControls(**cfg_dict)
    assert base_cfg.initial_precrack_wake_mode == InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE
    finite_cfg = base_cfg.validate()
    zero_cfg = replace(base_cfg, restored_work_of_separation_J_m2=0.0).validate()
    return finite_cfg, zero_cfg


def _stats_at_cursor(cfg, wave, ctrl, cursor_s: float) -> dict:
    Engine.configure_hazard(mode="exponential", seed=1720)
    Engine.reset_audit()
    engine, _audit = build_a_native_engine(cfg)
    assert len(engine._rebonding_state.active) == 0, "test precondition: zero active patches"
    engine._rebonding_state.elapsed_time_s = cursor_s
    return _phase_statistics(engine, ctrl, wave, 300.0)


@pytest.mark.parametrize("cursor_frac", [0.0, 0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.99])
def test_zero_patches_finite_and_zero_cohesion_are_identical_at_any_cursor_with_dwell(cursor_frac):
    finite_cfg, zero_cfg = _finite_and_zero_configs()
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.002)
    ctrl = FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=80, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )
    cursor_s = cursor_frac * wave.period_s
    stats_finite = _stats_at_cursor(finite_cfg, wave, ctrl, cursor_s)
    stats_zero = _stats_at_cursor(zero_cfg, wave, ctrl, cursor_s)
    for key in stats_finite:
        assert stats_finite[key] == pytest.approx(stats_zero[key], rel=1.0e-10, abs=1.0e-30), (
            f"{key} diverged at cursor_frac={cursor_frac}: finite={stats_finite[key]!r} zero={stats_zero[key]!r}"
        )


def test_zero_hold_behavior_unaffected_by_the_fix():
    """hold=0 must remain exactly as before: dt_values/dt_cleave_values are
    the same array by construction there, so this is a pure no-op check."""
    finite_cfg, zero_cfg = _finite_and_zero_configs()
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0)
    ctrl = FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=80, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )
    for frac in (0.0, 0.3, 0.7):
        cursor_s = frac * wave.period_s
        stats_finite = _stats_at_cursor(finite_cfg, wave, ctrl, cursor_s)
        stats_zero = _stats_at_cursor(zero_cfg, wave, ctrl, cursor_s)
        assert stats_finite["lambda_avg_s"] == pytest.approx(stats_zero["lambda_avg_s"], rel=1.0e-9)
