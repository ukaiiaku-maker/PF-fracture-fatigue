from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.persistent_site_reversible_transport_v10230 import (
    advect_signed_mobile_populations,
    signed_arrhenius_rate,
    signed_finite_tip_stress_Pa,
)
from arrhenius_fracture.unified_mpz import UnifiedMPZState


class Surface:
    @staticmethod
    def rate(stress, _temperature):
        return np.exp(np.asarray(stress, dtype=float) / 10.0)


def actual_empty_mpz():
    state = object.__new__(UnifiedMPZState)
    state.n_systems = 1
    state.n_bins = 4
    state.dx = 1.0
    state.mobile_negative = np.zeros((1, 4))
    state.mobile_positive = np.zeros((1, 4))
    state.mobile = np.zeros((1, 4))
    return state


def test_signed_rate_is_odd_and_exactly_zero_without_drive():
    rates = signed_arrhenius_rate(Surface(), np.array([-5.0, 0.0, 5.0]), 300.0)
    assert rates[0] == pytest.approx(-rates[2])
    assert rates[1] == 0.0


def test_positive_K_forward_limit_preserves_finite_tip_mapping():
    expected = 12.0e6 / np.sqrt(2.0 * np.pi * 1.0e-6)
    assert signed_finite_tip_stress_Pa(12.0e6, 1.0e-6) == pytest.approx(expected)
    assert signed_finite_tip_stress_Pa(-12.0e6, 1.0e-6) == pytest.approx(-expected)


def test_actual_mpz_signed_advection_distinguishes_return_and_escape():
    state = actual_empty_mpz()
    state.mobile_positive[0, 0] = 2.0
    state.mobile_negative[0, -1] = 3.0
    state.mobile = state.mobile_positive + state.mobile_negative
    fate = advect_signed_mobile_populations(
        state,
        velocity_by_system_m_s=np.array([-1.0]),
        dt_s=1.0,
        emitted_sign_by_system=np.array([1.0]),
        true_reverse_drive_by_system=np.array([True]),
    )
    assert fate["returned_mobile"][0, 1] == pytest.approx(2.0)
    assert fate["physical_returned_mobile"][0, 1] == pytest.approx(2.0)
    assert fate["escaped_mobile"][0, 0] == pytest.approx(3.0)
    assert np.sum(state.mobile) == 0.0


def test_raw_left_outflow_is_not_physical_return_without_reverse_drive():
    state = actual_empty_mpz()
    state.mobile_positive[0, 0] = 2.0
    fate = advect_signed_mobile_populations(
        state,
        np.array([-1.0]),
        1.0,
        np.array([1.0]),
        np.array([False]),
    )
    assert fate["returned_mobile"][0, 1] == pytest.approx(2.0)
    assert fate["physical_returned_mobile"][0, 1] == 0.0
