"""V2-B/Section 8: zero-cohesion RB2 must match RB0 event timing exactly,
across multiple events, while still evolving real (nonzero) P/C/B state.

Complements test_v10_2_30_crack_rebonding_contact_only_parity.py (RB0 vs
RB1) with the analogous RB0 vs RB2-zero-cohesion claim: a config with
model_level=CLEAN_REVERSIBLE_REBOND (kinetically active -- bonds can
genuinely form and rupture) but restored_work_of_separation_J_m2=0.0 (so
K_rebond_max is exactly 0 regardless of state) must not perturb cleavage
event timing at all, because crack_rebonding_v10230.cohesion_present(cfg)
gates the coupled event-time root-finder and the phase-shifted hazard
sampling off, exactly as it does for RB1 -- see
persistent_site_cyclic_energy_gated_v10230.py::_commit_rebonding_event and
persistent_site_coupled_hazard_v10229.py::_phase_statistics's
hazard_coupled branch.
"""
from __future__ import annotations

import dataclasses

import pytest

import _crack_rebonding_engine_fixture as fx

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import RebondModelLevel
from arrhenius_fracture.fatigue_v1 import FatigueWaveform

# Unit-test-only driving frequency, not a mission/pilot protocol choice: the
# DBTT_A0003837 test-fixture candidate's native cleavage hazard at the
# fixture's default f=1000 Hz fires far too fast for any patch to see a
# compressive excursion (the same "reference protocol may not sample
# compression" question the real pilot answers separately for A_NATIVE --
# see docs/v10_2_30_crack_rebonding_causal_pilot_v2.md). This value is
# chosen solely to demonstrate the zero-cohesion parity + live-kinetics
# mechanism through the real engine at unit-test scope.
_TEST_FREQUENCY_HZ = 1.0e6


def _zero_cohesion_cfg():
    cfg = fx.rebonding_cfg(model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND)
    return dataclasses.replace(cfg, restored_work_of_separation_J_m2=0.0)


def _test_waveform():
    return FatigueWaveform(Kmax=18.0e6, R=-0.95, frequency_Hz=_TEST_FREQUENCY_HZ)


def _drive_n_events(engine, waveform, n_events, *, max_blocks_per_event: int = 5000):
    events = []
    ctrl = fx.controller()
    for _ in range(n_events):
        fx.run_to_next_fired_event(engine, ctrl, waveform, max_blocks=max_blocks_per_event)
        length = fx.commit_pending_event(engine)
        rebonding_state = getattr(engine, "_rebonding_state", None)
        max_pB = (
            max((float(p.p_B) for p in rebonding_state.active), default=0.0)
            if rebonding_state is not None
            else 0.0
        )
        events.append(
            {
                "length_m": length,
                "elapsed_time_s": (
                    float(rebonding_state.elapsed_time_s) if rebonding_state is not None else None
                ),
                "max_pB": max_pB,
            }
        )
    return events


def test_zero_cohesion_rb2_matches_rb0_event_lengths_across_multiple_events():
    N_EVENTS = 4

    Engine = fx.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    Engine.configure_hazard(mode="exponential", seed=99)
    Engine.reset_audit()
    engine_rb0 = fx.build_real_engine(None)
    waveform = _test_waveform()
    events_rb0 = _drive_n_events(engine_rb0, waveform, N_EVENTS)

    Engine.configure_hazard(mode="exponential", seed=99)
    Engine.reset_audit()
    engine_rb2_zero = fx.build_real_engine(_zero_cohesion_cfg())
    events_rb2_zero = _drive_n_events(engine_rb2_zero, waveform, N_EVENTS)

    for i, (e0, e2) in enumerate(zip(events_rb0, events_rb2_zero)):
        assert e2["length_m"] == pytest.approx(e0["length_m"], rel=1.0e-9), (
            f"event {i}: zero-cohesion RB2 length diverged from RB0"
        )

    # The Markov state must still be genuinely, dynamically evolving despite
    # zero cleavage-hazard coupling -- otherwise this "parity" would be
    # trivially true for the wrong reason (patch_Q never running at all).
    assert any(e["max_pB"] > 0.0 for e in events_rb2_zero), (
        "zero-cohesion RB2 patches never bonded -- kinetics should still be "
        "active even though K_rebond_max is forced to zero"
    )


def test_zero_cohesion_rb2_K_rebond_is_exactly_zero_despite_nonzero_bonding():
    Engine = fx.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    Engine.configure_hazard(mode="exponential", seed=99)
    Engine.reset_audit()
    engine = fx.build_real_engine(_zero_cohesion_cfg())
    waveform = _test_waveform()
    ctrl = fx.controller()
    fx.run_to_next_fired_event(engine, ctrl, waveform)
    fx.commit_pending_event(engine)

    rebonding_state = engine._rebonding_state
    for _ in range(2000):
        result = engine.cycle_step_waveform(ctrl, waveform, 300.0)
        assert rebonding_state.K_rebond_Pa_sqrt_m == 0.0
        if result.get("fired"):
            break
