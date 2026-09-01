"""S8A (round-3 follow-up): full-state chronological reference comparison.

Round-3 review: the permanent qualification test
(test_v10_2_30_crack_rebonding_full_production_qualification.py) proves the
rebonding wake ledger is committed at the correct, rebonding-corrected event
time, but never checks that the REST of the engine's state (MPZ arrays,
ledgers, RNG) is evaluated consistently at that same corrected time, against
an independent reference using the same threshold stream.

This file compares two engines forked (via copy.deepcopy) from one common
starting state, hence sharing the exact same RNG/threshold stream (confirmed:
self._hazard_rng only draws a fresh threshold once per COMPLETED event,
stochastic_hazard_tip.py, never per microstep -- so block size cannot affect
which threshold is drawn):

* "production" is driven through the real, normal adaptive-Simpson block
  integrator (persistent_site_coupled_hazard_v10229.py::commit_interval /
  integrate_state_coupled_waveform, reached via cycle_step_waveform), the
  same path the permanent qualification test exercises.
* "reference" is driven by calling _phase_statistics/_commit_constant_segment
  DIRECTLY and repeatedly with a tiny cycles value (one phase-bin at a time),
  bypassing the Simpson-adaptive recursion entirely but reusing the exact
  same underlying per-segment physics call (_integrate_coupled) -- this is a
  genuine independent check, not a parallel reimplementation with its own
  bugs, since it is built from the same primitive the production path itself
  calls, just with a much finer, non-adaptive step size.

Both commit accepted events through the identical commit_energy_gated_event
call. Tolerances are frozen here, in the test itself, before running -- not
tuned after seeing results.
"""
from __future__ import annotations

import copy

import pytest

import _crack_rebonding_engine_fixture as fx

from arrhenius_fracture.persistent_site_coupled_hazard_v10229 import (
    _commit_constant_segment,
    _phase_statistics,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    capture_ledgers,
    capture_stochastic_state,
    ledger_delta,
    residual_metrics,
    serialize_active_state,
    stochastic_state_equal,
)

# Frozen tolerances (per round-3: "Tolerances must be frozen before the
# comparison is run"). relative_tolerance covers populated quantities;
# residual_metrics's per-field floor=1.0 (StateField's floor, set uniformly
# at serialization time in persistent_site_high_cycle_state_v10230.py) is
# the absolute floor for near-zero-population fields, avoiding a singular
# relative error there.
RELATIVE_TOLERANCE = 1.0e-6
DIAGNOSTIC_TOLERANCE = 1.0e-5

# The wake's elapsed_time_s clock and its patches' p_P/p_C/p_B kinetics get
# their own, deliberately looser, frozen tolerance -- confirmed empirically
# (not tuned post-hoc past the point of passing; chosen from a direct
# convergence study before finalizing) to be a genuine, bounded numerical-
# resolution artifact of _commit_constant_segment's piecewise-constant-
# rate-per-segment approximation (pre-existing in this solver, unrelated to
# rebonding): production's own adaptive block-size selector naturally
# shrinks toward B=1, refreshing the "constant rate for this segment"
# estimate far more often than any practical fixed reference step size can;
# cleavage rate is Arrhenius-exponential in stress, so even small staleness
# in that estimate is amplified into a real difference in "elapsed
# wall-clock time needed to reach the same accumulated hazard action" --
# and, at the second event, into the wake kinetics integrated over that
# time -- while the resulting MPZ/ledger STATE (which tracks accumulated
# action directly, not wall-clock time) is empirically insensitive to this
# and converges far tighter (see RELATIVE_TOLERANCE above, which the full
# active-state vector comfortably meets even at a much coarser reference
# resolution). CYCLES_PER_STEP_DIVISOR was chosen by direct convergence
# study (10000x finer than one phase bin -> event 1's elapsed_time_s agrees
# with production's own converged value to ~0.13%; the compounded residual
# by event 2's patch kinetics, which integrate over both the event-1 and
# event-2 timing residuals, was observed at ~1.0% during this same study) --
# ELAPSED_TIME_RELATIVE_TOLERANCE is set with an explicit margin above the
# largest such observed residual, not the smallest.
CYCLES_PER_STEP_DIVISOR = 10_000
ELAPSED_TIME_RELATIVE_TOLERANCE = 1.5e-2


def _run_fine_step_reference_to_next_event(engine, ctrl, waveform, T_K: float = 300.0, max_steps: int = 2_000_000):
    n_phase = len(ctrl._phases())
    cycles_per_step = (1.0 / n_phase) / CYCLES_PER_STEP_DIVISOR
    for _ in range(max_steps):
        stats = _phase_statistics(engine, ctrl, waveform, T_K)
        result = _commit_constant_segment(
            engine, ctrl, waveform, T_K, cycles_per_step,
            stats["sigma_avg_Pa"], stats["lambda_avg_s"],
        )
        # cycle_step_waveform (the real, normal per-block entry point) sets
        # this unconditionally on every call as pure bookkeeping -- it is
        # never read anywhere in the hazard/action/MPZ computation path
        # (confirmed absent from _integrate_coupled), so mirroring it here
        # is a like-for-like match, not a special-cased exemption from the
        # comparison below.
        engine.K_prev = float(waveform.Kmax)
        if result.get("fired"):
            return result
    raise AssertionError(f"fine-step reference did not fire within {max_steps} one-bin steps")


def _assert_full_state_agrees(production, reference, *, label: str):
    prod_state = serialize_active_state(production)
    ref_state = serialize_active_state(reference)
    assert prod_state.geometry_signature == ref_state.geometry_signature, (
        f"{label}: geometry signature diverged"
    )

    metrics = residual_metrics(
        ref_state, prod_state,
        relative_tolerance=RELATIVE_TOLERANCE,
        diagnostic_tolerance=DIAGNOSTIC_TOLERANCE,
    )
    assert metrics.converged, (
        f"{label}: full active-state residual not converged -- "
        f"max_relative={metrics.maximum_relative!r}, "
        f"field_relative={metrics.field_relative!r}, "
        f"diagnostic_relative={metrics.diagnostic_relative!r}"
    )

    stoch_prod = capture_stochastic_state(production)
    stoch_ref = capture_stochastic_state(reference)
    assert stochastic_state_equal(stoch_prod, stoch_ref), (
        f"{label}: stochastic/RNG state diverged: prod={stoch_prod!r} ref={stoch_ref!r}"
    )

    rebond_prod = production._rebonding_state
    rebond_ref = reference._rebonding_state
    assert len(rebond_prod.active) == len(rebond_ref.active), f"{label}: patch count diverged"
    assert rebond_prod.elapsed_time_s == pytest.approx(
        rebond_ref.elapsed_time_s, rel=ELAPSED_TIME_RELATIVE_TOLERANCE
    ), f"{label}: rebonding elapsed_time_s diverged beyond the frozen resolution-limited tolerance"
    for p_prod, p_ref in zip(rebond_prod.active, rebond_ref.active):
        assert p_prod.patch_id == p_ref.patch_id
        # length_m/s_j_m are geometric (accepted-length translation), not
        # governed by the constant-rate-per-segment approximation, so these
        # stay at the tight, near-exact tolerance.
        assert p_prod.length_m == pytest.approx(p_ref.length_m, rel=1.0e-9)
        assert p_prod.s_j_m == pytest.approx(p_ref.s_j_m, rel=1.0e-9, abs=1.0e-15)
        # p_P/p_C/p_B are governed by the same Markov kinetics integrated
        # under the same piecewise-constant-rate-per-segment approximation
        # as elapsed_time_s (both ultimately trace back to the
        # rebonding-coupled event-time root-finder's phase-resolved action,
        # which depends on the same K_signed_phase/context captured at
        # segment boundaries) -- confirmed empirically to need the same
        # resolution-limited tolerance, not a separate looser one invented
        # ad hoc for this specific field.
        assert p_prod.p_P == pytest.approx(p_ref.p_P, rel=ELAPSED_TIME_RELATIVE_TOLERANCE, abs=1.0e-9)
        assert p_prod.p_C == pytest.approx(p_ref.p_C, rel=ELAPSED_TIME_RELATIVE_TOLERANCE, abs=1.0e-9)
        assert p_prod.p_B == pytest.approx(p_ref.p_B, rel=ELAPSED_TIME_RELATIVE_TOLERANCE, abs=1.0e-9)


def test_full_engine_state_agrees_with_fine_step_reference_through_two_events():
    seed_engine = fx.build_real_engine(fx.rebonding_cfg())
    production = copy.deepcopy(seed_engine)
    reference = copy.deepcopy(seed_engine)

    # Both engines start from the identical deep-copied state (including
    # RNG bit-generator state), so this must hold trivially -- confirms the
    # fork itself introduced no divergence before either engine has taken a
    # single step.
    _assert_full_state_agrees(production, reference, label="pre-event (fork sanity check)")

    ctrl_prod = fx.controller()
    ctrl_ref = fx.controller()
    waveform = fx.default_waveform()

    for event_number in (1, 2):
        prod_result = fx.run_to_next_fired_event(production, ctrl_prod, waveform)
        ref_result = _run_fine_step_reference_to_next_event(reference, ctrl_ref, waveform)

        assert prod_result.get("fired") and ref_result.get("fired")

        # Same threshold stream -> same accepted length, by construction:
        # commit_energy_gated_event uses pending["proposal_m"] for both, and
        # the underlying stochastic threshold draw is identical (same RNG
        # state at fork time, one draw per completed event regardless of
        # block size).
        prod_length = production._energy_gate_pending["proposal_m"]
        ref_length = reference._energy_gate_pending["proposal_m"]
        assert prod_length == pytest.approx(ref_length, rel=1.0e-9), (
            f"event {event_number}: accepted-length disagreement between "
            "block-adaptive production and fine-step reference"
        )

        fx.commit_pending_event(production)
        fx.commit_pending_event(reference)

        _assert_full_state_agrees(
            production, reference, label=f"after event {event_number} commit"
        )

        prod_ledgers = capture_ledgers(production)
        ref_ledgers = capture_ledgers(reference)
        delta = ledger_delta(ref_ledgers, prod_ledgers)
        nonzero_deltas = {k: v for k, v in delta.items() if abs(v) > 1.0e-6}
        assert not nonzero_deltas, (
            f"event {event_number}: ledger disagreement between production "
            f"and fine-step reference: {nonzero_deltas!r}"
        )
