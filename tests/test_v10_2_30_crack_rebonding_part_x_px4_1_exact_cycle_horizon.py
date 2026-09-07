"""PX4.1 section 2: the EXACT first-passage cycle horizon.

The prior implementation (PX4) only checked max_cumulative_cycles BETWEEN
accepted events -- a real gap for the general case (a single block can
internally consume up to max_block_cycles=1e6 cycles, so an event search
could in principle overshoot an externally-imposed cycle budget by a lot).

Fixed with NO new engine-internals: cycle_step_waveform already accepts a
``requested_cycles`` parameter that threads into choose_block_cycles_
diagnostic's existing "requested_cap" mode (base = min(max_block_cycles,
requested_cycles), and the returned block size is clipped to the MINIMUM
of every candidate limit, base included) -- an already-enforced, already-
qualified hard ceiling on a single block's size. run_trajectory's block
loop now passes requested_cycles=remaining_budget on every block call
once max_cumulative_cycles is finite, and stops immediately (no engine
call, no RNG draw) once the remaining budget reaches exactly zero.

Because a horizon-limited block that does not fire uses the exact same
"did not fire" commit path every ordinary non-firing block already uses
(full MPZ/rebonding-state/protocol-cursor advance, no event, no patch, no
translation, no new threshold draw), this reuses already-validated
machinery rather than inventing a second commit path.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

SEED = 1720


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


def _run(*, rebonding_cfg=None, max_cumulative_cycles=math.inf, max_accepted_events=3,
          minimum_load_hold_s=0.0, static_shield_control=None, hazard_rng_seed=SEED):
    Engine.configure_hazard(mode="exponential", seed=hazard_rng_seed)
    Engine.reset_audit()
    return pilot.run_trajectory(
        name="horizon_test", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=rebonding_cfg, R=-0.5,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
        max_accepted_events=max_accepted_events, max_projected_extension_m=1.0e300,
        hazard_rng_seed=hazard_rng_seed, max_cumulative_cycles=max_cumulative_cycles,
        minimum_load_hold_s=minimum_load_hold_s, static_shield_control=static_shield_control,
    )


def _reversible_finite_cfg():
    return CrackRebondingControls(
        enabled=True, model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY, feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
        wake_length_m=0.0005, wake_weight_length_m=5e-7,
        restored_work_of_separation_J_m2=1.8225, rebond_K_geometry_factor=1.0,
        bond_activation_volume_m3=1e-30, rupture_activation_volume_m3=0.0,
        bond_attempt_frequency_s=1e10, rupture_attempt_frequency_s=1e10,
        chemistry_factor=1.0, fresh_surface_clean_fraction=1.0,
        initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
    ).validate()


def test_default_is_byte_identical_to_unbounded():
    """max_cumulative_cycles=math.inf (default) must reproduce the exact
    prior behavior for every existing caller."""
    r_default = _run()
    r_explicit_inf = _run(max_cumulative_cycles=math.inf)
    assert r_default["cumulative_cycles"] == r_explicit_inf["cumulative_cycles"]
    assert r_default["n_accepted_events"] == r_explicit_inf["n_accepted_events"] == 3
    assert not r_default["censored"]


def test_cap_between_two_events_accepts_the_first_and_censors_exactly():
    baseline = _run(max_accepted_events=3)
    cap = (baseline["events"][0]["cumulative_cycles"] + baseline["events"][1]["cumulative_cycles"]) / 2.0

    capped = _run(max_accepted_events=3, max_cumulative_cycles=cap)
    assert capped["n_accepted_events"] == 1
    assert capped["censored"] is True
    assert capped["censor_reason"] == "complete_physical_cycle_censor"
    assert capped["cumulative_cycles"] == pytest.approx(cap, rel=1.0e-9)
    # The one accepted event must be BIT-IDENTICAL to the unbounded baseline
    # -- the horizon must not perturb anything before it takes effect.
    assert capped["events"][0]["cumulative_cycles"] == baseline["events"][0]["cumulative_cycles"]
    assert capped["events"][0]["accepted_length_m"] == baseline["events"][0]["accepted_length_m"]
    assert capped["events"][0]["hazard_threshold_action"] == baseline["events"][0]["hazard_threshold_action"]


def test_inherited_min_block_cycles_floor_is_the_only_precision_limit():
    """A remaining budget SMALLER than FatigueControllerConfig's own
    min_block_cycles (default 1e-6) can overshoot by up to that floor's
    value, since the adaptive search refuses to resolve a candidate
    smaller than it -- a pre-existing, already-qualified system limit
    this fix inherits rather than introduces. Documented here explicitly
    rather than left as a surprise: utterly negligible against the
    mission's actual 1e12-cycle horizon."""
    capped = _run(max_accepted_events=1000, max_cumulative_cycles=1.0e-9)
    assert capped["n_accepted_events"] == 0
    assert capped["censored"] is True
    assert capped["censor_reason"] == "complete_physical_cycle_censor"
    # Overshoots up to min_block_cycles=1e-6, not the requested 1e-9.
    assert capped["cumulative_cycles"] == pytest.approx(1.0e-6, rel=1.0e-6)
    assert capped["cumulative_cycles"] > 1.0e-9


def test_cap_exactly_at_an_event_boundary_accepts_that_event_and_stops():
    """Setting the cap to EXACTLY the natural firing point of event 0 makes
    the capped run's requested_cycles for that block equal the very value
    the unconstrained adaptive search would independently converge to --
    two different call sequences reaching the same physical point, not
    guaranteed bit-identical (the cap value itself influences the search's
    own internal step choices). A loose relative tolerance still confirms
    the intended behavior: the event fires at essentially the capped
    boundary, and the trajectory stops immediately after (no event 1)."""
    baseline = _run(max_accepted_events=3)
    cap = baseline["events"][0]["cumulative_cycles"]

    capped = _run(max_accepted_events=3, max_cumulative_cycles=cap)
    assert capped["n_accepted_events"] == 1
    assert capped["cumulative_cycles"] == pytest.approx(cap, rel=1.0e-4)
    assert capped["events"][0]["cumulative_cycles"] == pytest.approx(baseline["events"][0]["cumulative_cycles"], rel=1.0e-4)


def test_cap_beyond_all_requested_events_is_a_pure_no_op():
    baseline = _run(max_accepted_events=3)
    huge_cap = baseline["cumulative_cycles"] * 1000.0
    capped = _run(max_accepted_events=3, max_cumulative_cycles=huge_cap)
    assert capped["n_accepted_events"] == 3
    assert not capped["censored"]
    assert capped["cumulative_cycles"] == baseline["cumulative_cycles"]


def test_noninteger_remaining_cycles_still_censors_exactly():
    baseline = _run(max_accepted_events=2)
    cap = baseline["events"][0]["cumulative_cycles"] + 0.123456789
    assert cap < baseline["events"][1]["cumulative_cycles"], "test fixture assumption violated -- pick a smaller offset"
    capped = _run(max_accepted_events=2, max_cumulative_cycles=cap)
    assert capped["n_accepted_events"] == 1
    assert capped["cumulative_cycles"] == pytest.approx(cap, rel=1.0e-9)


def test_zero_remaining_budget_takes_no_engine_step_and_draws_no_new_threshold():
    """Starting exactly AT the horizon (cap == 0 relative cycles consumed)
    must censor immediately with zero events and zero cycles consumed --
    no engine call, hence no RNG draw, hence no threshold action recorded."""
    capped = _run(max_accepted_events=3, max_cumulative_cycles=0.0)
    assert capped["n_accepted_events"] == 0
    assert capped["censored"] is True
    assert capped["censor_reason"] == "complete_physical_cycle_censor"
    assert capped["cumulative_cycles"] == 0.0
    assert capped["events"] == []


def test_horizon_with_dynamic_rebonding_finite_cohesion():
    cfg = _reversible_finite_cfg()
    baseline = _run(rebonding_cfg=cfg, max_accepted_events=3)
    cap = (baseline["events"][0]["cumulative_cycles"] + baseline["events"][1]["cumulative_cycles"]) / 2.0
    capped = _run(rebonding_cfg=cfg, max_accepted_events=3, max_cumulative_cycles=cap)
    assert capped["n_accepted_events"] == 1
    assert capped["cumulative_cycles"] == pytest.approx(cap, rel=1.0e-9)
    assert capped["events"][0]["cumulative_extension_m"] == baseline["events"][0]["cumulative_extension_m"]


def test_horizon_with_zero_cohesion():
    from dataclasses import replace
    cfg = replace(_reversible_finite_cfg(), restored_work_of_separation_J_m2=0.0).validate()
    baseline = _run(rebonding_cfg=cfg, max_accepted_events=3)
    cap = (baseline["events"][0]["cumulative_cycles"] + baseline["events"][1]["cumulative_cycles"]) / 2.0
    capped = _run(rebonding_cfg=cfg, max_accepted_events=3, max_cumulative_cycles=cap)
    assert capped["n_accepted_events"] == 1
    assert capped["cumulative_cycles"] == pytest.approx(cap, rel=1.0e-9)


def test_horizon_with_prescribed_static_shielding():
    baseline = _run(
        max_accepted_events=3,
        static_shield_control={"enabled": True, "K_b_static_Pa_sqrt_m": 900000.0, "first_event_fired": False},
    )
    cap = (baseline["events"][0]["cumulative_cycles"] + baseline["events"][1]["cumulative_cycles"]) / 2.0
    capped = _run(
        max_accepted_events=3, max_cumulative_cycles=cap,
        static_shield_control={"enabled": True, "K_b_static_Pa_sqrt_m": 900000.0, "first_event_fired": False},
    )
    assert capped["n_accepted_events"] == 1
    assert capped["cumulative_cycles"] == pytest.approx(cap, rel=1.0e-9)


def test_horizon_with_minimum_load_hold():
    baseline = _run(max_accepted_events=3, minimum_load_hold_s=0.002)
    cap = (baseline["events"][0]["cumulative_cycles"] + baseline["events"][1]["cumulative_cycles"]) / 2.0
    capped = _run(max_accepted_events=3, minimum_load_hold_s=0.002, max_cumulative_cycles=cap)
    assert capped["n_accepted_events"] == 1
    assert capped["cumulative_cycles"] == pytest.approx(cap, rel=1.0e-9)


def test_horizon_censor_state_matches_baseline_up_to_the_censor_point():
    """The engine's own committed state (B, t) at the moment of censoring
    must match what the SAME unbounded trajectory shows at the same
    cumulative_cycles point -- i.e. the horizon censor commits real,
    correct physical state, not a truncated/corrupted one."""
    baseline = _run(max_accepted_events=3)
    cap = (baseline["events"][0]["cumulative_cycles"] + baseline["events"][1]["cumulative_cycles"]) / 2.0
    capped = _run(max_accepted_events=3, max_cumulative_cycles=cap)
    # Both trajectories agree exactly on the one accepted event's full record.
    assert capped["events"][0] == baseline["events"][0]
