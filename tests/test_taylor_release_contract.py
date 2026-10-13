import numpy as np

from arrhenius_fracture.taylor_release import (
    NATIVE_MEAN_MODE,
    completion_rate,
    density_completion_order,
    historical_legacy_implicit_1s,
    runtime_selection,
    signed_arrhenius_components,
)
from arrhenius_fracture.material_manifest import ExpFloorBarrier
from arrhenius_fracture.unified_mpz import UnifiedMPZState


def test_historical_one_second_expression_is_quarantined_and_reproducible():
    rate = np.array([0.0, 1.0e-20, 0.25, 1.0, 1.0e6, 1.0e13])
    order = np.array([1.0, 2.0, 3.0, 7.5, 50.0, 2.0])
    historical = historical_legacy_implicit_1s(rate, order)
    native = completion_rate(rate, order)
    assert np.all(historical <= 1.0)
    assert native[-1] == 5.0e12
    assert runtime_selection() == (NATIVE_MEAN_MODE, None, None)


def test_native_mean_completion_is_dimensionally_unit_invariant_and_uncapped():
    rate_s = np.array([1.0e-6, 2.5e4, 1.0e14])
    order = np.array([1.0, 4.5, 43.0])
    per_s = completion_rate(rate_s, order)
    per_ms = completion_rate(rate_s / 1000.0, order)
    np.testing.assert_allclose(per_ms * 1000.0, per_s, rtol=2e-15)
    assert per_s[-1] > 1.0e12


def test_density_order_is_the_documented_uncapped_expression():
    rho = np.array([0.0, 4.0e12, 1.0e24])
    actual = density_completion_order(
        rho, correlation_scale=2.5, correlation_density_m2=1.0e12
    )
    np.testing.assert_allclose(actual, 1.0 + 2.5 * np.sqrt(rho / 1.0e12))
    assert actual[-1] > 1.0e6


def test_forward_reverse_identity_zero_bias_and_reversal():
    surface = ExpFloorBarrier(1.2, 0.0, 2.0e9, 0.0, 1.4, 1.1, 0.1)
    stress = np.array([-8.0e8, 0.0, 8.0e8])
    parts = signed_arrhenius_components(surface, stress, 950.0)
    assert parts["net_rate_s"][1] == 0.0
    assert parts["forward_rate_s"][1] == parts["reverse_rate_s"][1]
    assert parts["net_rate_s"][0] == -parts["net_rate_s"][2]
    assert parts["forward_rate_s"][0] == parts["reverse_rate_s"][2]
    assert parts["reverse_rate_s"][0] == parts["forward_rate_s"][2]


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
