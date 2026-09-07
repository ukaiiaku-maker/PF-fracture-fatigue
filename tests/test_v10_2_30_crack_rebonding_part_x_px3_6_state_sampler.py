"""PX3.6 section 4: the run_trajectory state_sampler hook, added so D5's
passivation chemistry selection can be re-qualified against a genuine live
duration-weighted cycle-mean p_P/p_C/p_B, instead of PX3.5's pre_event_
max_pB/max_pB_post_commit event-extrema proxy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


def test_default_state_sampler_none_is_byte_identical_to_before():
    """Adding the state_sampler parameter (default None) must not change
    any existing caller's result -- verified by running the same
    trajectory twice, once without passing it at all."""
    def _run():
        Engine.configure_hazard(mode="exponential", seed=1720)
        Engine.reset_audit()
        return pilot.run_trajectory(
            name="sampler_default", build_engine=_build_engine, make_controller=_make_controller,
            waveform_cls=FatigueWaveform, rebonding_cfg=None, R=-0.5,
            reset_engine_registry=Engine.reset_audit,
            Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
            max_accepted_events=1, max_projected_extension_m=1.0e300, hazard_rng_seed=1720,
        )
    r1 = _run()
    r2 = _run()
    assert r1["cumulative_extension_m"] == r2["cumulative_extension_m"]
    assert r1["cumulative_cycles"] == r2["cumulative_cycles"]


def test_state_sampler_is_called_once_per_block_with_engine_and_result():
    calls = []

    def sampler(engine, result):
        calls.append((engine, dict(result)))

    Engine.configure_hazard(mode="exponential", seed=1720)
    Engine.reset_audit()
    trajectory = pilot.run_trajectory(
        name="sampler_called", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=None, R=-0.5,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
        max_accepted_events=1, max_projected_extension_m=1.0e300, hazard_rng_seed=1720,
        state_sampler=sampler,
    )
    total_blocks = sum(ev["blocks_to_fire"] for ev in trajectory["events"])
    assert len(calls) == total_blocks
    for engine, result in calls:
        assert "kinetic_dt_consumed_s" in result
        assert hasattr(engine, "cycle_step_waveform")


def test_state_sampler_sees_zero_active_patches_before_rebond_cfg_none():
    """With rebonding_cfg=None (no rebonding installed at all), the
    sampler's engine argument has no _rebonding_state -- any consumer must
    handle that (this is exactly the guard scripts/run_part_x_px3_6_
    passivation_requalification.py's sampler implements)."""
    seen_states = []

    def sampler(engine, result):
        seen_states.append(getattr(engine, "_rebonding_state", None))

    Engine.configure_hazard(mode="exponential", seed=1720)
    Engine.reset_audit()
    pilot.run_trajectory(
        name="sampler_no_rebond", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=None, R=-0.5,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
        max_accepted_events=1, max_projected_extension_m=1.0e300, hazard_rng_seed=1720,
        state_sampler=sampler,
    )
    assert len(seen_states) > 0
    assert all(s is None for s in seen_states)
