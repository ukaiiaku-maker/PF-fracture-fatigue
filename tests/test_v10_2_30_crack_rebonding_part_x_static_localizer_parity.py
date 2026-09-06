"""PX1.4 (Part X): prescribed-static event-localization parity.

Generalizes the exact phase-resolved event-time evaluator so a
prescribed CONSTANT K_b (the static-shield mechanism-control ablation)
gets access to the identical localization semantics dynamic rebonding
uses (exact bisection via solve_coupled_event_time against a genuine
phase-resolved action), rather than the coarser cycle-mean adaptive
quadrature the existing static-shield-attribution study used. Mission
section 5.4's frozen validation: compare the new exact evaluator against
an INDEPENDENT fine-stepped reference across starting phases, B_start
values, Kmax, K_b, firing inside a partial cycle, and firing inside the
minimum-load hold -- tolerance frozen below BEFORE any comparison is
run.

Tolerance frozen before running: relative action agreement 1e-6 (both
methods are evaluating the exact same piecewise-constant-K(phase)
discretized model -- the fine reference substeps within that same
discretization very finely, so the only expected discrepancy is
floating-point roundoff and the fine reference's own residual
substep-linearization error, not a genuine numerical-method difference).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_v10230 import (
    cleavage_stress_with_rebond,
    solve_coupled_event_time,
    static_shield_phase_resolved_action,
)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform

FROZEN_ACTION_REL_TOL = 1.0e-6


def _lambda_cleave_fn(sigma_Pa: float) -> float:
    """A representative monotone cleavage-rate law (arbitrary but fixed
    for this qualification -- the mission's own gate is on the
    EVALUATOR's numerical fidelity, not on any particular physical rate
    law)."""
    return 1.0e-4 * max(sigma_Pa, 0.0) ** 1.2


def _fine_stepped_reference_action(
    k0, t_interval, K_phase_fn, dt_array, K_shield, K_b_static, r_eff, lambda_fn, substeps_per_bin=500,
):
    """Independent brute-force reference: for each bin the constant-K_b
    action is simply lambda_cleave(sigma_c) * duration (sigma_c does not
    vary within a bin since K_phase_fn/K_b_static are both constant
    there), so a genuinely independent method is to substep EACH bin at
    much finer resolution than the production bin count and sum -- this
    exercises the phase-schedule construction/bin-walking logic
    completely independently of static_shield_phase_resolved_action's own
    control flow."""
    n = len(dt_array)
    idx = k0 % n
    t_left = t_interval
    action = 0.0
    while t_left > 1e-15:
        bin_dt = dt_array[idx]
        take = min(bin_dt, t_left)
        K_s = K_phase_fn(idx)
        sigma_c = cleavage_stress_with_rebond(K_s, K_shield, K_b_static, r_eff)
        sub_dt = take / substeps_per_bin
        for _ in range(substeps_per_bin):
            action += lambda_fn(sigma_c) * sub_dt
        t_left -= take
        if take >= bin_dt - 1e-12:
            idx = (idx + 1) % n
    return action


def _schedule_for(Kmax, R, hold_s, n_phase=16):
    wave = FatigueWaveform(Kmax=Kmax, R=R, frequency_Hz=1000.0, minimum_load_hold_s=hold_s)
    K_signed, dt_values = wave.cycle_schedule(n_phase, signed=True)
    return K_signed, dt_values


@pytest.mark.parametrize("Kmax", [15.0e6, 18.0e6, 21.0e6])
@pytest.mark.parametrize("K_b_static", [0.45e6, 0.9e6, 1.8e6])
@pytest.mark.parametrize("k0", [0, 3, 8, 15])
@pytest.mark.parametrize(
    "t_interval_fraction",
    [0.3, 1.7, 8.3],  # partial cycle, spans a full cycle, spans several cycles
)
def test_static_action_matches_fine_stepped_reference_no_hold(Kmax, K_b_static, k0, t_interval_fraction):
    K_signed, dt_values = _schedule_for(Kmax, R=-0.5, hold_s=0.0)
    n_phase = len(dt_values)

    def K_phase_fn(idx):
        return float(K_signed[idx % n_phase])

    t_interval = t_interval_fraction * float(dt_values.sum())
    action, end_states, end_idx = static_shield_phase_resolved_action(
        k0=k0, t_interval=t_interval, K_phase_fn=K_phase_fn, dt_phase=dt_values,
        n_phase=n_phase, K_shield_Pa_sqrt_m=0.0, K_b_static=K_b_static, r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
    )
    ref_action = _fine_stepped_reference_action(
        k0, t_interval, K_phase_fn, dt_values, 0.0, K_b_static, 1.0e-6, _lambda_cleave_fn,
    )
    assert end_states == {}
    rel_err = abs(action - ref_action) / max(abs(ref_action), 1e-300)
    assert rel_err <= FROZEN_ACTION_REL_TOL


@pytest.mark.parametrize("Kmax", [15.0e6, 18.0e6, 21.0e6])
@pytest.mark.parametrize("K_b_static", [0.45e6, 0.9e6, 1.8e6])
def test_static_action_matches_fine_stepped_reference_firing_inside_hold(Kmax, K_b_static):
    """Firing partway through the (long) dwell bin -- the case the
    mission specifically calls out."""
    hold_s = 0.002  # much longer than one sinusoidal bin, at f=1000Hz/n_phase=16
    K_signed, dt_values = _schedule_for(Kmax, R=-0.5, hold_s=hold_s)
    n_phase = len(dt_values)
    hold_bin_idx = n_phase - 1
    assert dt_values[hold_bin_idx] == pytest.approx(hold_s)

    def K_phase_fn(idx):
        return float(K_signed[idx % n_phase])

    # Start exactly at the hold bin, stop partway through it.
    t_interval = 0.35 * hold_s
    action, end_states, end_idx = static_shield_phase_resolved_action(
        k0=hold_bin_idx, t_interval=t_interval, K_phase_fn=K_phase_fn, dt_phase=dt_values,
        n_phase=n_phase, K_shield_Pa_sqrt_m=0.0, K_b_static=K_b_static, r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
    )
    ref_action = _fine_stepped_reference_action(
        hold_bin_idx, t_interval, K_phase_fn, dt_values, 0.0, K_b_static, 1.0e-6, _lambda_cleave_fn,
    )
    assert end_idx == hold_bin_idx  # did not consume the whole (long) bin
    rel_err = abs(action - ref_action) / max(abs(ref_action), 1e-300)
    assert rel_err <= FROZEN_ACTION_REL_TOL


def test_one_cycle_action_identical_every_cycle_no_state_to_decay():
    """Unlike dynamic rebonding, a prescribed constant K_b carries no
    state from cycle to cycle -- the action contributed by cycle N and
    cycle N+1, starting from the same phase index, must be EXACTLY
    (bit-for-bit) equal, with no periodic-orbit convergence/transient
    concept applicable at all."""
    K_signed, dt_values = _schedule_for(18.0e6, R=-0.5, hold_s=0.0005)
    n_phase = len(dt_values)

    def K_phase_fn(idx):
        return float(K_signed[idx % n_phase])

    total_cycle = float(dt_values.sum())
    action_1_cycle, _, idx_1 = static_shield_phase_resolved_action(
        k0=0, t_interval=total_cycle, K_phase_fn=K_phase_fn, dt_phase=dt_values,
        n_phase=n_phase, K_shield_Pa_sqrt_m=0.0, K_b_static=0.9e6, r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
    )
    action_2_cycles, _, idx_2 = static_shield_phase_resolved_action(
        k0=0, t_interval=2.0 * total_cycle, K_phase_fn=K_phase_fn, dt_phase=dt_values,
        n_phase=n_phase, K_shield_Pa_sqrt_m=0.0, K_b_static=0.9e6, r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
    )
    assert action_2_cycles == pytest.approx(2.0 * action_1_cycle, rel=1e-12)
    assert idx_1 == idx_2 == 0


# ---------------------------------------------------------------------------
# solve_coupled_event_time integration: exact event-time root-finding for
# the static-shield mechanism, using the identical bisection dynamic
# rebonding uses.
# ---------------------------------------------------------------------------


def test_solve_coupled_event_time_localizes_static_event_correctly():
    """End-to-end: root-find the exact firing time for a prescribed
    constant K_b using solve_coupled_event_time (unmodified -- it is
    already fully generic in its phase_resolved_action_fn argument, so
    PX1.4 required zero changes to it), then confirm the converged action
    matches an independent direct evaluation at that exact dt."""
    K_signed, dt_values = _schedule_for(18.0e6, R=-0.5, hold_s=0.0005)
    n_phase = len(dt_values)

    def K_phase_fn(idx):
        return float(K_signed[idx % n_phase])

    K_b_static = 0.9e6
    B_start = 0.3
    B_threshold = 1.0

    def phase_resolved_action_fn(dt):
        return static_shield_phase_resolved_action(
            k0=0, t_interval=dt, K_phase_fn=K_phase_fn, dt_phase=dt_values,
            n_phase=n_phase, K_shield_Pa_sqrt_m=0.0, K_b_static=K_b_static, r_eff_m=1.0e-6,
            lambda_cleave_fn=_lambda_cleave_fn,
        )

    total_cycle = float(dt_values.sum())
    # A rough constant-rate estimate for the uncoupled bracket seed.
    rough_action, _, _ = phase_resolved_action_fn(total_cycle)
    lambda_avg_uncoupled = rough_action / total_cycle

    def integrate_coupled_fn(lambda_avg):
        if lambda_avg <= 0.0:
            return {"fired": False}
        return {"fired": True, "dt_consumed": (B_threshold - B_start) / lambda_avg}

    root = solve_coupled_event_time(
        integrate_coupled_fn=integrate_coupled_fn,
        phase_resolved_action_fn=phase_resolved_action_fn,
        lambda_avg_uncoupled=lambda_avg_uncoupled,
        B_start=B_start, B_threshold=B_threshold, eps_B=1.0e-9,
        dt_block=total_cycle,
    )
    assert root["fired"] is True
    assert root["converged"] is True
    action_at_dt_used, _, _ = phase_resolved_action_fn(root["dt_used"])
    assert B_start + action_at_dt_used == pytest.approx(B_threshold, abs=1e-8)


def test_zero_K_b_reduces_to_ordinary_zero_shield_action():
    """Architectural note tested directly: K_b_static=0 is mathematically
    a valid degenerate input to this function (cleavage_stress_with_rebond
    simply subtracts nothing), matching the ordinary no-shield sigma_c --
    though per the mission, callers must not ROUTE K_b=0 through this
    exact-bisection path in production (the cheap original baseline must
    be used instead); this only confirms the function itself does not
    silently misbehave if it were ever called with K_b=0."""
    K_signed, dt_values = _schedule_for(18.0e6, R=-0.5, hold_s=0.0)
    n_phase = len(dt_values)

    def K_phase_fn(idx):
        return float(K_signed[idx % n_phase])

    action_zero_Kb, _, _ = static_shield_phase_resolved_action(
        k0=0, t_interval=float(dt_values.sum()), K_phase_fn=K_phase_fn, dt_phase=dt_values,
        n_phase=n_phase, K_shield_Pa_sqrt_m=0.0, K_b_static=0.0, r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
    )
    ref_action = _fine_stepped_reference_action(
        0, float(dt_values.sum()), K_phase_fn, dt_values, 0.0, 0.0, 1.0e-6, _lambda_cleave_fn,
    )
    rel_err = abs(action_zero_Kb - ref_action) / max(abs(ref_action), 1e-300)
    assert rel_err <= FROZEN_ACTION_REL_TOL
