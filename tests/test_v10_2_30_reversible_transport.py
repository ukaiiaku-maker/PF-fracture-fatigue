from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.persistent_site_reversible_transport_v10230 import (
    advect_signed_mobile_populations,
    install_reversible_transport,
    net_source_slip,
    signed_arrhenius_rate,
    signed_finite_tip_stress_Pa,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    apply_ledger_delta,
    capture_ledgers,
    restore_active_state,
    serialize_active_state,
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


class LinearSurface:
    @staticmethod
    def rate(stress, _temperature):
        return 1.0 + 1.0e-10 * np.asarray(stress, dtype=float)


class ZeroSurface:
    @staticmethod
    def rate(stress, _temperature):
        return np.zeros_like(np.asarray(stress, dtype=float))


def evolvable_actual_mpz():
    state = actual_empty_mpz()
    state.x = (np.arange(4) + 0.5) * state.dx
    state.wake_n_bins = 4
    state.wake_dx = state.dx
    state.wake_mobile_positive = np.zeros((1, 4))
    state.wake_mobile_negative = np.zeros((1, 4))
    state.wake_retained_positive = np.zeros((1, 4))
    state.wake_retained_negative = np.zeros((1, 4))
    state.wake_mobile = np.zeros((1, 4))
    state.wake_retained = np.zeros((1, 4))
    state.retained_positive = np.zeros((1, 4))
    state.retained_negative = np.zeros((1, 4))
    state.retained = np.zeros((1, 4))
    state.accumulated_slip_positive = np.zeros((1, 4))
    state.accumulated_slip_negative = np.zeros((1, 4))
    state.accumulated_slip = np.zeros((1, 4))
    state.site_capacity = np.ones(1)
    state.available_sites = np.ones(1)
    state.cfg = SimpleNamespace(
        blunting_length_m=state.dx,
        source_bin_count=1,
        forest_density_floor_m2=1.0,
        peierls_stress_fraction=1.0,
        taylor_stress_fraction=1.0,
        jump_fraction=1.0,
        mobile_recovery_rate_s=0.0,
    )
    state.manifest = SimpleNamespace(
        c_blunt=2.0,
        encounter_efficiency=0.0,
        retained_recovery_rate_s=0.0,
        taylor_corr_rho_c_m2=1.0,
        taylor_corr_scale=0.0,
        peierls=SimpleNamespace(as_surface=lambda _emission: LinearSurface()),
        taylor=SimpleNamespace(as_surface=lambda _emission: ZeroSurface()),
        emission=SimpleNamespace(),
    )
    state._signed_transport_mode = "validated_scalar"
    state.signed_last_source_activations = 0.0
    state.signed_last_burgers_sign_by_system = np.array([1.0])
    state.escaped_total = 0.0
    state.recovered_total = 0.0
    state.time_s = 0.0
    state._transport_rates = MethodType(UnifiedMPZState._transport_rates, state)
    state.local_stress_profile_Pa = MethodType(UnifiedMPZState.local_stress_profile_Pa, state)
    state.local_forest_density_m2 = MethodType(UnifiedMPZState.local_forest_density_m2, state)
    def emit(self, _dt, opening, _temperature, _weights=None):
        self.last_opening_emission_stress_Pa = opening
        return 0.0

    state._emit = MethodType(emit, state)
    install_reversible_transport(state)
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
        source_linked_eligible_by_system_sign=np.array([[False, True]]),
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
        np.array([[False, False]]),
    )
    assert fate["returned_mobile"][0, 1] == pytest.approx(2.0)
    assert fate["physical_returned_mobile"][0, 1] == 0.0


def test_installed_actual_state_reverses_mobile_and_bounds_net_blunting():
    state = evolvable_actual_mpz()
    state.mobile_positive[0, 0] = 2.0
    state.accumulated_slip_positive[0, 0] = 1.25
    state.mobile = state.mobile_positive.copy()
    state.accumulated_slip = state.accumulated_slip_positive.copy()
    state._reversible_transport_K_signed_Pa_sqrt_m = -1.0e6
    state._reversible_tip_radius_m = 1.0e-6
    before_radius = state.blunted_radius(1.0e-6, 1.0e-10)
    result = state.evolve(1.0e-3, 300.0, 0.0, 1.0e-10)
    assert state.last_opening_emission_stress_Pa == 0.0
    assert result["dN_physical_returned"] > 0.0
    assert result["dN_source_slip_cancelled"] == pytest.approx(
        min(result["dN_physical_returned"], 1.25)
    )
    assert np.sum(state.returned_slip_positive) <= 1.25
    assert np.all(net_source_slip(state) >= 0.0)
    assert state.blunted_radius(1.0e-6, 1.0e-10) <= before_radius


def test_return_does_not_cancel_retained_population():
    state = evolvable_actual_mpz()
    state.mobile_positive[0, 0] = 1.0
    state.retained_positive[0, 0] = 7.0
    state.accumulated_slip_positive[0, 0] = 1.0
    state.mobile = state.mobile_positive.copy()
    state.retained = state.retained_positive.copy()
    state.accumulated_slip = state.accumulated_slip_positive.copy()
    state._reversible_transport_K_signed_Pa_sqrt_m = -1.0e6
    state._reversible_tip_radius_m = 1.0e-6
    state.evolve(1.0e-3, 300.0, 0.0, 1.0e-10)
    assert np.sum(state.retained_positive) == pytest.approx(7.0)


def test_reversible_active_state_and_rate_separated_ledgers_round_trip():
    state = evolvable_actual_mpz()
    engine = SimpleNamespace(
        mpz=state,
        W_emit=0.0,
        K_prev=0.0,
        n_adv=0,
        a_adv=0.0,
        micro_advance_total_m=0.0,
        checkpoint_advance_total_m=0.0,
        r_eff=lambda: state.blunted_radius(1.0e-6, 1.0e-10),
        K_shield=lambda: 0.0,
    )
    state.returned_slip_positive[0, 0] = 0.25
    state.cumulative_returned_mobile[0, 1] = 2.0
    snapshot = serialize_active_state(engine)
    names = {field.name for field in snapshot.fields}
    assert "returned_slip_positive" in names
    state.returned_slip_positive[:] = 0.0
    restore_active_state(engine, snapshot)
    assert state.returned_slip_positive[0, 0] == pytest.approx(0.25)
    ledgers = capture_ledgers(engine)
    key = "mpz.cumulative_returned_mobile[0,1]"
    assert ledgers[key] == pytest.approx(2.0)
    apply_ledger_delta(engine, {key: 3.0})
    assert state.cumulative_returned_mobile[0, 1] == pytest.approx(5.0)
