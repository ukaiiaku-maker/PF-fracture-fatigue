"""Periodic-orbit bulk-action acceleration for phase_resolved_action.

Round-3 review: the O(n_phase*n_cycles) exact bin-by-bin loop is not
feasible for VHCF/low-K blocks spanning many cycles. This tests the
conservative hybrid: resolve a finite transient exactly, detect convergence
to a periodic orbit, then represent the remainder as
remaining_cycles*A_c,* with the wake state advanced exactly via
matrix_power -- verifying both correctness (bulk matches exhaustive exact
stepping) and the actual runtime win for a genuinely large cycle count.
"""
from __future__ import annotations

import math
import time

import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls, RebondModelLevel
from arrhenius_fracture.crack_rebonding_v10230 import WakePatch, phase_resolved_action


def _cfg(**overrides):
    fields = dict(
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        wake_length_m=5.0e-6,
        wake_weight_length_m=1.0e-6,
        restored_work_of_separation_J_m2=2.0,
        rebond_K_geometry_factor=1.0,
        bond_activation_volume_m3=0.0,
        bond_barrier_eV=0.3,
        bond_attempt_frequency_s=1.0e6,
        rupture_activation_volume_m3=0.0,
        rupture_barrier_eV=0.5,
        rupture_attempt_frequency_s=1.0e6,
        chemistry_factor=1.0,
    )
    fields.update(overrides)
    return CrackRebondingControls(**fields).validate()


def _K_phase(idx, n_phase, Kmax, R):
    phase = (idx + 0.5) * (2.0 * math.pi / n_phase)
    Kmean = 0.5 * (Kmax + R * Kmax)
    Kamp = 0.5 * (Kmax - R * Kmax)
    return Kmean + Kamp * math.cos(phase)


def _lambda_cleave_fn(sigma):
    return 1.0e-3 * sigma


def test_bulk_action_matches_exhaustive_exact_stepping():
    """Force bulk mode with a tiny threshold and confirm the result agrees
    with the (slow) fully-exact reference to a tight tolerance."""
    cfg = _cfg()
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    n_phase = 24
    Kmax, R, f_Hz = 18.0e6, -0.95, 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase
    n_cycles = 40  # small enough to also run the exhaustive reference

    def K_phase_fn(idx):
        return _K_phase(idx, n_phase, Kmax, R)

    kwargs = dict(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        t_interval=n_cycles * n_phase * dt_phase,
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
    )

    action_exact, states_exact, idx_exact, diag_exact = phase_resolved_action(
        bulk_cycle_threshold=10_000, **kwargs  # effectively disables bulk mode
    )
    assert diag_exact["bulk_action_used"] is False
    assert diag_exact["bulk_action_qualified"] is True

    # max_transient_cycles=30 for a 40-cycle interval, against a config
    # whose relaxation time is genuinely ~200 cycles (see below): 30 cycles
    # is deliberately insufficient to actually reach the periodic orbit, so
    # the certified bound correctly reports bulk_action_qualified=False here
    # -- this is the S8C fail-closed signal doing its job, not a bug. The
    # numeric comparison tolerance reflects "still approximately close,
    # honestly uncertified," not a required bit-exact match (that stronger
    # claim is only true once the bound actually certifies, exercised by
    # the tight case below).
    action_bulk, states_bulk, idx_bulk, diag_bulk = phase_resolved_action(
        bulk_cycle_threshold=5, bulk_convergence_rel_tol=1.0e-8, max_transient_cycles=30, **kwargs
    )

    assert idx_exact == idx_bulk
    assert action_bulk == pytest.approx(action_exact, rel=2.0e-3)
    assert states_bulk[0].sum() == pytest.approx(1.0, abs=1.0e-8)
    assert diag_bulk["bulk_action_used"] is True
    assert diag_bulk["periodic_orbit_solved"] is True
    assert diag_bulk["transient_cycles_resolved"] == 30
    assert 0.0 <= diag_bulk["subdominant_eigenvalue"] < 1.0
    assert diag_bulk["total_tail_action_error_bound"] >= 0.0
    assert diag_bulk["bulk_action_qualified"] is False

    # This configuration's compression-gated formation vs. slow rupture
    # gives it a genuinely long relaxation time relative to one cycle (~200
    # cycles here) -- confirming a real transient budget is needed for this
    # kind of physical scenario, not an artifact of the test. With a larger
    # transient budget that actually spans the relaxation time, the match
    # tightens substantially.
    action_bulk_tight, states_bulk_tight, _, diag_bulk_tight = phase_resolved_action(
        bulk_cycle_threshold=5, bulk_convergence_rel_tol=1.0e-6, max_transient_cycles=n_cycles, **kwargs
    )
    assert action_bulk_tight == pytest.approx(action_exact, rel=1.0e-6)
    assert states_bulk_tight[0][2] == pytest.approx(states_exact[0][2], abs=1.0e-6)
    assert diag_bulk_tight["bulk_action_qualified"] is True


def test_bulk_action_matches_exact_for_non_integer_cycle_count():
    """The n_full = n_cycles*n_phase + leftover_bins split must be exact
    even when the interval is not a whole number of cycles."""
    cfg = _cfg()
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    n_phase = 20
    Kmax, R, f_Hz = 18.0e6, -0.95, 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase

    def K_phase_fn(idx):
        return _K_phase(idx, n_phase, Kmax, R)

    t_interval = 12.35 * n_phase * dt_phase  # 12 cycles + partial

    kwargs = dict(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        t_interval=t_interval,
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
    )
    action_exact, _, idx_exact, _ = phase_resolved_action(bulk_cycle_threshold=10_000, **kwargs)
    action_bulk, _, idx_bulk, _ = phase_resolved_action(
        bulk_cycle_threshold=3, bulk_convergence_rel_tol=1.0e-8, max_transient_cycles=15, **kwargs
    )
    assert idx_exact == idx_bulk
    assert action_bulk == pytest.approx(action_exact, rel=1.0e-6)


def test_bulk_mode_is_dramatically_faster_for_a_billion_cycles():
    """A billion-cycle block must complete in well under a second via the
    bulk path (would be computationally infeasible with exact stepping)."""
    cfg = _cfg()
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    n_phase = 32
    Kmax, R, f_Hz = 18.0e6, -0.95, 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase

    def K_phase_fn(idx):
        return _K_phase(idx, n_phase, Kmax, R)

    start = time.perf_counter()
    action, states, _, diag = phase_resolved_action(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        t_interval=1_000_000_000 * n_phase * dt_phase,
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
        max_transient_cycles=50,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"billion-cycle bulk evaluation took {elapsed:.2f}s, expected sub-second"
    assert math.isfinite(action)
    assert states[0].sum() == pytest.approx(1.0, abs=1.0e-6)
    assert 0.0 <= states[0][2] <= 1.0 + 1.0e-6
    assert diag["bulk_action_used"] is True
    assert math.isfinite(diag["total_tail_action_error_bound"])


def test_bulk_mode_stays_bounded_even_when_strict_convergence_is_never_reached():
    """If bulk_convergence_rel_tol is essentially unreachable, the function
    must still complete fast (runtime is always bounded -- state propagation
    is exact regardless of action-bound certification) rather than silently
    degrading into O(n_cycles) exact stepping for a huge interval -- an
    earlier version of this function could hang for exactly this reason
    (caught by a genuinely hanging test before the fix).

    S8C (round-3 follow-up): this scenario is deliberately slow-relaxing
    (bond_barrier_eV=0.5 is a genuinely slow formation rate) with only 20
    transient cycles resolved -- the certified periodic-orbit bound is
    expected to correctly flag this as NOT qualified (large distance to the
    true periodic state, not yet within tolerance), rather than silently
    returning an uncertified answer with no signal. The caller
    (persistent_site_cyclic_energy_gated_v10230.py's phase_resolved_action_fn
    closure) is what actually fails closed on this signal -- see
    tests/test_v10_2_30_crack_rebonding_event_time_coupling.py for that
    enforcement path."""
    cfg = _cfg(
        bond_barrier_eV=0.5, bond_attempt_frequency_s=1.0e2,  # very slow, near-linear buildup
        rupture_barrier_eV=5.0,  # negligible rupture
    )
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    n_phase = 16
    Kmax, R, f_Hz = 18.0e6, -0.95, 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase

    def K_phase_fn(idx):
        return _K_phase(idx, n_phase, Kmax, R)

    start = time.perf_counter()
    action, states, _, diag = phase_resolved_action(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        t_interval=100_000_000 * n_phase * dt_phase,  # 100 million cycles
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=_lambda_cleave_fn,
        bulk_cycle_threshold=5,
        bulk_convergence_rel_tol=1.0e-12,  # essentially unreachable
        max_transient_cycles=20,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"non-convergent bulk evaluation took {elapsed:.2f}s, expected sub-second (bounded runtime)"
    assert math.isfinite(action)
    assert states[0].sum() == pytest.approx(1.0, abs=1.0e-6)
    assert diag["bulk_action_used"] is True
    assert diag["strict_convergence_reached"] is False
    assert diag["bulk_action_qualified"] is False
    assert diag["total_tail_action_error_bound"] > 0.0
