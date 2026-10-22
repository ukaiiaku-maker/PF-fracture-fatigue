import math
import numpy as np

from scripts.refit_qste340tm_capacity_policy import exact_retained_step, retained_source_per_cycle


def test_uncapped_exact_solution_and_zero_recovery_limit():
    n, source, recovery = exact_retained_step(3.0, 0.02, 0.0, 1234.0, None)
    assert n == 3.0 + 0.02 * 1234.0
    assert source == 0.02 * 1234.0
    assert recovery == 0.0


def test_uncapped_exact_solution_with_recovery():
    n0, j, gamma, cycles = 3.0, 0.02, 1e-4, 1234.0
    n, source, recovery = exact_retained_step(n0, j, gamma, cycles, None)
    expected = j/gamma + (n0-j/gamma)*math.exp(-gamma*cycles)
    np.testing.assert_allclose(n, expected, rtol=2e-15)
    np.testing.assert_allclose(n-n0, source-recovery, rtol=2e-14)


def test_fixed_capacity_balance_and_bound():
    n, source, recovery = exact_retained_step(3.0, 0.2, 2e-5, 1e6, 1000.0)
    assert 0 <= n < 1000.0
    np.testing.assert_allclose(n-3.0, source-recovery, rtol=2e-12, atol=1e-12)


def test_birth_phase_retention_exceeds_full_cycle_for_positive_escape():
    emit = np.ones(8) * 2.0
    escape = np.ones(8) * 5.0
    _, _, native = retained_source_per_cycle(emit, escape, 10.0, "native_full_cycle")
    _, _, resolved = retained_source_per_cycle(emit, escape, 10.0, "birth_phase_resolved")
    assert resolved > native > 0.0


def test_continuous_birth_constant_coefficients_are_exact():
    for panels in (1, 2, 8, 32, 128):
        emit = np.full(panels, 2.0)
        escape = np.full(panels, 5.0)
        _, _, got = retained_source_per_cycle(emit, escape, 10.0, "continuous_birth_exact")
        expected = 2.0 / 5.0 * (1.0 - math.exp(-0.5))
        np.testing.assert_allclose(got, expected, rtol=2e-15)


def test_continuous_birth_zero_limits_and_survivor_bounds():
    _, _, zero_source = retained_source_per_cycle(np.zeros(4), np.ones(4), 2.0, "continuous_birth_exact")
    _, _, zero_escape = retained_source_per_cycle(np.ones(4), np.zeros(4), 2.0, "continuous_birth_exact")
    assert zero_source == 0.0
    assert zero_escape == 0.5
    _, _, late = retained_source_per_cycle(np.array([0.0, 0.0, 0.0, 4.0]), np.full(4, 100.0), 1.0, "continuous_birth_exact")
    assert 0.0 < late < 1.0


def test_event_free_restart_matches_one_frozen_coefficient_step():
    initial, source, recovery = 37.0, 2.5e-4, 7.0e-5
    first, admitted_1, recovered_1 = exact_retained_step(
        initial, source, recovery, 1234.5, None
    )
    restarted, admitted_2, recovered_2 = exact_retained_step(
        first, source, recovery, 8765.5, None
    )
    direct, admitted, recovered = exact_retained_step(
        initial, source, recovery, 10000.0, None
    )
    np.testing.assert_allclose(restarted, direct, rtol=2e-14)
    np.testing.assert_allclose(admitted_1 + admitted_2, admitted, rtol=2e-14)
    np.testing.assert_allclose(recovered_1 + recovered_2, recovered, rtol=2e-14)
