import numpy as np
from scipy.special import gammainc

from arrhenius_fracture.taylor_release import (
    FINITE_WINDOW_MODE,
    LEGACY_MODE,
    SEQUENTIAL_MEAN_MODE,
    completion_rate,
    finite_window_completion,
)
from arrhenius_fracture.unified_mpz import UnifiedMPZState


def test_legacy_is_exact_archived_expression():
    rate = np.array([0.0, 1.0e-20, 0.25, 1.0, 1.0e6, 1.0e13])
    order = np.array([1.0, 2.0, 3.0, 7.5, 50.0, 2.0])
    expected = gammainc(np.maximum(order, 1.0), np.clip(rate, 0.0, 1.0e12))
    actual = completion_rate(rate, order, mode=LEGACY_MODE)
    assert np.array_equal(actual, expected)
    assert np.all(actual <= 1.0)


def test_finite_window_changes_argument_and_normalization():
    rate = np.array([1.0e2, 1.0e8, 1.0e11])
    order = np.array([1.0, 3.0, 43.0])
    tau = 1.0e-9
    expected = gammainc(order, np.minimum(rate, 1.0e12) * tau) / tau
    actual = completion_rate(rate, order, mode=FINITE_WINDOW_MODE, window_s=tau)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert not np.allclose(actual, completion_rate(rate, order, mode=LEGACY_MODE) / tau)


def test_seconds_to_milliseconds_unit_invariance():
    rate_s = np.array([1.0e-6, 2.5e4, 1.0e11])
    order = np.array([1.0, 4.5, 43.0])
    per_s = finite_window_completion(
        rate_s, order, window_in_unit=1.0e-9, rate_cap_per_unit=1.0e12
    )
    rate_ms = rate_s / 1000.0
    per_ms = finite_window_completion(
        rate_ms, order, window_in_unit=1.0e-6, rate_cap_per_unit=1.0e9
    )
    np.testing.assert_allclose(per_ms * 1000.0, per_s, rtol=2.0e-15, atol=0.0)


def test_limits_and_nonfinite_inputs():
    rate = np.array([0.0, -1.0, np.nan, np.inf, 1.0e-30, 1.0e12])
    order = np.array([1.0, 1.0, 1.0, 1.0, 8.0, 1.0e6])
    for mode in (LEGACY_MODE, FINITE_WINDOW_MODE, SEQUENTIAL_MEAN_MODE):
        result = completion_rate(rate, order, mode=mode, window_s=1.0e-9)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert result[0] == result[1] == result[2] == 0.0
    first_order = completion_rate(
        np.array([1.0e-12]), np.array([1.0]), mode=FINITE_WINDOW_MODE, window_s=1.0e-9
    )
    np.testing.assert_allclose(first_order, [1.0e-12], rtol=2.0e-15)


def test_exchange_conserves_inventory_and_positivity_for_stiff_release():
    mobile = np.array([[2.0, 0.0], [3.0, 4.0]])
    retained = np.array([[5.0, 7.0], [0.0, 11.0]])
    before = mobile + retained
    new_m, new_r, trapped, released = UnifiedMPZState._exchange(
        mobile,
        retained,
        np.array([2.5e12, 4.0e3]),
        np.array([1.0e9, 7.0e11]),
        8.4,
    )
    np.testing.assert_allclose(new_m + new_r, before, rtol=0.0, atol=2.0e-15)
    assert np.all(new_m >= 0.0)
    assert np.all(new_r >= 0.0)
    assert trapped >= 0.0 and released >= 0.0


def test_exact_exchange_is_timestep_partition_invariant_at_fixed_coefficients():
    mobile = np.array([[2.0, 1.0]])
    retained = np.array([[8.0, 9.0]])
    encounter = np.array([2.5e12, 4.0e3])
    release = np.array([1.0e9, 7.0e11])
    whole = UnifiedMPZState._exchange(mobile, retained, encounter, release, 1.0e-8)[:2]
    half = UnifiedMPZState._exchange(mobile, retained, encounter, release, 0.5e-8)[:2]
    split = UnifiedMPZState._exchange(half[0], half[1], encounter, release, 0.5e-8)[:2]
    np.testing.assert_allclose(split[0], whole[0], rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(split[1], whole[1], rtol=2.0e-15, atol=2.0e-15)
