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
