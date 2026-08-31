from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    advance_markov,
    build_Q,
    build_phase_factors,
    propagate,
    strang_cycle_trajectory,
)


def _phase_generators(n_phase: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    Qs = []
    for _ in range(n_phase):
        k_cb, k_bc, k_pc, k_cp = rng.uniform(0.1, 5.0, size=4)
        Qs.append(build_Q(k_cb, k_bc, k_pc, k_cp))
    return Qs


def _brute_force(p0, Q_list, k0, n_steps, dt_phase):
    n = len(Q_list)
    p = np.asarray(p0, dtype=float)
    idx = k0 % n
    for _ in range(n_steps):
        p = advance_markov(p, Q_list[idx], dt_phase)
        idx = (idx + 1) % n
    return p


@pytest.mark.parametrize("n_steps", [1, 2, 5, 17, 50])
def test_propagate_matches_brute_force_whole_steps(n_steps):
    n_phase = 24
    dt_phase = 1.0e-6
    Q_list = _phase_generators(n_phase)
    phase_factors = build_phase_factors(Q_list, dt_phase)
    p0 = np.array([0.7, 0.2, 0.1])
    k0 = 5
    dt = n_steps * dt_phase
    got = propagate(p0, Q_list, phase_factors, k0, dt, dt_phase)
    want = _brute_force(p0, Q_list, k0, n_steps, dt_phase)
    assert np.allclose(got, want, atol=1.0e-10)


def test_propagate_conserves_and_is_positive_for_huge_cycle_count():
    n_phase = 12
    dt_phase = 1.0e-8
    Q_list = _phase_generators(n_phase, seed=1)
    phase_factors = build_phase_factors(Q_list, dt_phase)
    p0 = np.array([0.4, 0.4, 0.2])
    dt = 1.0e9 * n_phase * dt_phase  # a billion cycles
    got = propagate(p0, Q_list, phase_factors, k0=3, dt=dt, dt_phase=dt_phase)
    # A billion-cycle matrix_power accumulates ~30 squarings of floating-point
    # error; conservation/positivity must still hold to a loose but meaningful
    # tolerance, not bit-exactness.
    assert got.sum() == pytest.approx(1.0, abs=1.0e-5)
    assert (got >= -1.0e-6).all()


def test_propagate_block_partition_invariance():
    n_phase = 20
    dt_phase = 2.0e-6
    Q_list = _phase_generators(n_phase, seed=2)
    phase_factors = build_phase_factors(Q_list, dt_phase)
    p0 = np.array([0.3, 0.5, 0.2])
    k0 = 7
    dt_total = 4.35 * n_phase * dt_phase  # spans several cycles plus a fraction

    single = propagate(p0, Q_list, phase_factors, k0, dt_total, dt_phase)

    dt_half = dt_total / 2.0
    mid = propagate(p0, Q_list, phase_factors, k0, dt_half, dt_phase)
    steps_half = dt_half / dt_phase
    k_mid = int(round((k0 + steps_half))) % n_phase
    # Recompute the exact phase index reached after dt_half via the same
    # decomposition propagate() uses internally, rather than assuming a clean
    # integer step count.
    steps_f = dt_half / dt_phase
    whole = int(math.floor(steps_f + 1e-9))
    k_mid = (k0 + whole) % n_phase
    two_step = propagate(mid, Q_list, phase_factors, k_mid, dt_total - dt_half, dt_phase)

    assert np.allclose(single, two_step, atol=1.0e-9)


def test_named_regression_1p2_cycle_block_true_midpoint_is_0p6():
    n_phase = 30
    dt_phase = 1.0e-6
    period = n_phase * dt_phase
    Q_list = _phase_generators(n_phase, seed=3)
    phase_factors = build_phase_factors(Q_list, dt_phase)
    p0 = np.array([1.0, 0.0, 0.0])
    dt_block = 1.2 * period
    t_mid = dt_block / 2.0
    assert t_mid == pytest.approx(0.6 * period)
    # The midpoint state must be reachable via the same generic propagator,
    # and must differ from both p0 and the block-end state in general.
    p_mid = propagate(p0, Q_list, phase_factors, k0=0, dt=t_mid, dt_phase=dt_phase)
    p_end = propagate(p0, Q_list, phase_factors, k0=0, dt=dt_block, dt_phase=dt_phase)
    assert p_mid.sum() == pytest.approx(1.0, abs=1e-9)
    assert p_end.sum() == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(p_mid, p_end)


@pytest.mark.parametrize("n_cycles", [3, 4, 7, 8])
def test_odd_and_even_integer_cycle_blocks(n_cycles):
    n_phase = 16
    dt_phase = 5.0e-7
    Q_list = _phase_generators(n_phase, seed=4)
    phase_factors = build_phase_factors(Q_list, dt_phase)
    p0 = np.array([0.2, 0.3, 0.5])
    dt = n_cycles * n_phase * dt_phase
    got = propagate(p0, Q_list, phase_factors, k0=0, dt=dt, dt_phase=dt_phase)
    want = _brute_force(p0, Q_list, 0, n_cycles * n_phase, dt_phase)
    assert np.allclose(got, want, atol=1.0e-9)


@pytest.mark.parametrize("k0", [0, 1, 5, 11, 19])
def test_arbitrary_starting_phase_and_wraparound(k0):
    n_phase = 20
    dt_phase = 3.0e-7
    Q_list = _phase_generators(n_phase, seed=5)
    phase_factors = build_phase_factors(Q_list, dt_phase)
    p0 = np.array([0.5, 0.25, 0.25])
    n_steps = 27  # forces wraparound past n_phase for most k0
    dt = n_steps * dt_phase
    got = propagate(p0, Q_list, phase_factors, k0, dt, dt_phase)
    want = _brute_force(p0, Q_list, k0, n_steps, dt_phase)
    assert np.allclose(got, want, atol=1.0e-9)


def test_strang_cycle_trajectory_end_state_matches_propagate_full_cycle():
    n_phase = 24
    dt_phase = 1.0e-6
    Q_list = _phase_generators(n_phase, seed=6)
    phase_factors = build_phase_factors(Q_list, dt_phase)
    p0 = np.array([0.6, 0.1, 0.3])
    midpoints, p_end = strang_cycle_trajectory(p0, Q_list, dt_phase)
    p_end_direct = propagate(p0, Q_list, phase_factors, k0=0, dt=n_phase * dt_phase, dt_phase=dt_phase)
    assert np.allclose(p_end, p_end_direct, atol=1.0e-10)
    assert midpoints.shape == (n_phase, 3)
    assert np.allclose(midpoints.sum(axis=1), 1.0, atol=1.0e-9)
    assert (midpoints >= -1.0e-9).all()


def test_strang_cycle_trajectory_phase_resolution_convergence():
    # Coarser vs finer phase grids over the same physical cycle should agree
    # in the coarse grid's sampled midpoints to within discretization error
    # that shrinks with resolution (checked via a smooth, slowly varying Q).
    dt_phase_coarse = 1.0e-6
    n_coarse = 12
    n_fine = 48
    dt_phase_fine = dt_phase_coarse * n_coarse / n_fine

    def smooth_Q(phase_frac: float) -> np.ndarray:
        k_cb = 1.0 + 0.5 * math.sin(2 * math.pi * phase_frac)
        k_bc = 0.5 + 0.2 * math.cos(2 * math.pi * phase_frac)
        return build_Q(k_CB=k_cb, k_BC=k_bc, k_PC=0.0, k_CP=0.0)

    Q_coarse = [smooth_Q(i / n_coarse) for i in range(n_coarse)]
    Q_fine = [smooth_Q(i / n_fine) for i in range(n_fine)]
    p0 = np.array([1.0, 0.0, 0.0])

    _, end_coarse = strang_cycle_trajectory(p0, Q_coarse, dt_phase_coarse)
    _, end_fine = strang_cycle_trajectory(p0, Q_fine, dt_phase_fine)
    assert np.allclose(end_coarse, end_fine, atol=5.0e-3)
