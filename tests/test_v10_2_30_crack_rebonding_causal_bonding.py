"""S8B (round-3 follow-up): dynamically generated nonzero rebonding.

Round-3 review: the permanent qualification test's synthetic scenario fires
events within a fraction of a cycle -- too fast for a patch to actually pass
through a compression phase and develop p_B > 0. It therefore proves
transaction bookkeeping (accepted-length patch creation, no state
inheritance, rollback), but not that the LIVE kinetics generate a shielding
effect that changes the next event's waiting time:

    contact proxy -> bond formation -> K_rebond > 0 -> changed event time.

This file uses crack_rebonding_kinetics_v10230.solve_reference_action_barriers
with the "persistent" reference-action preset (A_on=10, A_off=0.1 --
formation-dominant, weak rupture) to calibrate a strong-but-nonsaturated
CLEAN_REVERSIBLE_REBOND configuration directly, rather than hand-tuning
barriers: this is exactly what that generator exists for. The specific
restored_work_of_separation_J_m2/rebond_K_geometry_factor pair used below
was chosen by a direct calibration sweep (not tuned post-hoc past the point
of passing): weaker values produce negligible, undetectable shielding;
markedly stronger values suppress the second event entirely within a
reasonable call budget (the saturated regime the mission explicitly asks
this pass to avoid). The chosen point produces order-40%-of-Kmax K_rebond
and roughly doubles the second event's waiting time relative to RB1 -- a
clear, non-saturated causal effect.
"""
from __future__ import annotations

import pytest

import _crack_rebonding_engine_fixture as fx

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    REFERENCE_ACTION_PRESETS,
    RebondModelLevel,
    solve_reference_action_barriers,
)


def _calibrated_rb2_cfg(reference_contact_radius_m: float):
    template = fx.rebonding_cfg(model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND)
    import dataclasses

    template = dataclasses.replace(
        template,
        restored_work_of_separation_J_m2=50.0,
        rebond_K_geometry_factor=2.0,
    )
    A_on_ref, A_off_ref = REFERENCE_ACTION_PRESETS["persistent"]
    return solve_reference_action_barriers(
        A_on_ref,
        A_off_ref,
        reference_patch_distance_m=0.0,
        reference_contact_radius_m=reference_contact_radius_m,
        cfg_template=template,
    )


def _drive_to_second_event(engine, waveform, *, max_blocks: int = 5000):
    """Fires event 1 and commits it, then drives toward event 2, sampling
    the first (event-1-created) patch's p_B and the wake's aggregate
    K_rebond after every block, returning (event2_result, max_pB, max_K_rebond)."""
    ctrl1 = fx.controller()
    fx.run_to_next_fired_event(engine, ctrl1, waveform)
    length1 = fx.commit_pending_event(engine)

    max_pB = 0.0
    max_K_rebond = 0.0
    ctrl2 = fx.controller()
    for _ in range(max_blocks):
        result = engine.cycle_step_waveform(ctrl2, waveform, 300.0)
        rebonding_state = getattr(engine, "_rebonding_state", None)
        if rebonding_state is not None:
            for patch in rebonding_state.active:
                max_pB = max(max_pB, patch.p_B)
            max_K_rebond = max(max_K_rebond, rebonding_state.K_rebond_Pa_sqrt_m)
        if result.get("fired"):
            return length1, result, max_pB, max_K_rebond
    raise AssertionError(f"second event did not fire within {max_blocks} blocks")


def test_live_kinetics_generate_nonzero_bonding_that_delays_the_next_event():
    seed_bare = fx.build_real_engine(None)
    reference_contact_radius_m = max(seed_bare.r_eff(), fx.rebonding_cfg().contact_radius_min_m)

    rb1_cfg = fx.rebonding_cfg(model_level=RebondModelLevel.CONTACT_PROXY_ONLY)
    rb2_cfg = _calibrated_rb2_cfg(reference_contact_radius_m)

    engine_rb1 = fx.build_real_engine(rb1_cfg)
    engine_rb2 = fx.build_real_engine(rb2_cfg)
    waveform = fx.default_waveform()

    length1_rb1, result2_rb1, _, _ = _drive_to_second_event(engine_rb1, waveform)
    length2_rb1 = fx.commit_pending_event(engine_rb1)

    length1_rb2, result2_rb2, max_pB, max_K_rebond = _drive_to_second_event(engine_rb2, waveform)

    # Causal claim: live kinetics on an event-created patch actually reach a
    # nonzero bonded fraction and a nonzero rebonding shielding contribution
    # before the second event -- not merely bookkeeping plumbing.
    assert max_pB > 0.0
    assert max_K_rebond > 0.0

    # Same threshold stream (both engines start fresh, same RNG seed policy)
    # -> event 1 must agree exactly between RB1 and RB2 regardless of the
    # rebonding model level, since no patch exists yet at event 1.
    assert length1_rb2 == pytest.approx(length1_rb1, rel=1.0e-9)

    t_event2_rb1 = engine_rb1._rebonding_state.elapsed_time_s
    length2_rb2 = fx.commit_pending_event(engine_rb2)
    t_event2_rb2 = engine_rb2._rebonding_state.elapsed_time_s

    # The causal effect itself: shielding from the live-formed bond delays
    # the second event relative to RB1 (contact-proxy-only, zero generator,
    # hence zero shielding) under the same threshold stream.
    assert t_event2_rb2 > t_event2_rb1

    # HAZARD_ONLY_REBOND_SHIELD invariants, confirmed on the real production
    # engine rather than assumed: the energy gate (accepted event length) is
    # never coupled to rebonding, in either mode, at either event.
    assert length2_rb2 == pytest.approx(length2_rb1, rel=1.0e-9)

    # The second event's newly created patch must still start completely
    # fresh (p_B=0), regardless of how much the first patch bonded.
    newest_rb2 = engine_rb2._rebonding_state.active[-1]
    assert newest_rb2.p_B == 0.0
    assert newest_rb2.s_j_m == pytest.approx(0.0)


def test_emission_is_bit_identical_between_rb1_and_rb2_despite_the_causal_shielding_effect():
    """HAZARD_ONLY_REBOND_SHIELD's core invariant: rebonding can delay first
    passage (confirmed above) but must never alter emission -- the sig array
    driving lam_e_site/mu_emit/stress_override is untouched at every one of
    the three injection points (see the equation-lineage doc)."""
    seed_bare = fx.build_real_engine(None)
    reference_contact_radius_m = max(seed_bare.r_eff(), fx.rebonding_cfg().contact_radius_min_m)

    rb1_cfg = fx.rebonding_cfg(model_level=RebondModelLevel.CONTACT_PROXY_ONLY)
    rb2_cfg = _calibrated_rb2_cfg(reference_contact_radius_m)

    engine_rb1 = fx.build_real_engine(rb1_cfg)
    engine_rb2 = fx.build_real_engine(rb2_cfg)
    waveform = fx.default_waveform()

    sig_rb1 = engine_rb1.sigma_tip(float(waveform.Kmax))
    sig_rb2 = engine_rb2.sigma_tip(float(waveform.Kmax))
    assert sig_rb2 == pytest.approx(sig_rb1, rel=1.0e-12)

    ctrl1, ctrl2 = fx.controller(), fx.controller()
    fx.run_to_next_fired_event(engine_rb1, ctrl1, waveform)
    fx.run_to_next_fired_event(engine_rb2, ctrl2, waveform)
    fx.commit_pending_event(engine_rb1)
    fx.commit_pending_event(engine_rb2)

    # After the first event (which seeds a real patch on RB2, whose
    # cleavage-side effect is nonzero per the test above), emission-facing
    # sigma_tip must still agree exactly -- rebonding never reaches this
    # call.
    sig_rb1_after = engine_rb1.sigma_tip(float(waveform.Kmax))
    sig_rb2_after = engine_rb2.sigma_tip(float(waveform.Kmax))
    assert sig_rb2_after == pytest.approx(sig_rb1_after, rel=1.0e-12)
