import math

import numpy as np
import pytest

from scripts.qste340tm_emission_blunting_reduction import (
    LOCAL_FORWARD_SLIP,
    LOCAL_SOURCE_SLIP,
    MOBILE,
    RETAINED,
    SOURCE_ACTIVITY,
    WAKE_MOBILE,
    WAKE_RETAINED,
    BluntingParameters,
    apply_cycles,
    convect_local_plasticity,
    effective_radius,
    phase_operator,
    population_backstress,
    resolve_increment_advection,
    wake_tip_projection,
)
from scripts.calibrate_qste340tm_stateful_exp_floor import BURGERS_M, TIP_RADIUS_M


def test_every_emitted_line_blunts_even_when_mobile_content_escapes():
    state = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    operator = phase_operator(
        hazard=10.0,
        clear=0.0,
        encounter=0.0,
        release=0.0,
        escape=1.0e12,
        retained_recovery=0.0,
        dt=1.0,
    )
    final = apply_cycles(operator, state, 1)
    assert final[MOBILE] == pytest.approx(0.0, abs=1.0e-12)
    assert final[RETAINED] == pytest.approx(0.0, abs=1.0e-12)
    assert final[LOCAL_SOURCE_SLIP] > 0.99
    assert final[LOCAL_FORWARD_SLIP] > 0.99
    assert effective_radius(final, BluntingParameters()) > TIP_RADIUS_M


def test_plastic_ledgers_do_not_manufacture_population_backstress():
    baseline = population_backstress(3.0, 7.0)
    state = np.array([3.0, 7.0, 1.0, 1.0e6, 2.0e6])
    assert population_backstress(state[MOBILE], state[RETAINED]) == baseline
    expected = TIP_RADIUS_M + BURGERS_M * 3.0e6
    assert effective_radius(state, BluntingParameters()) == pytest.approx(expected)


def test_forward_transport_is_recorded_separately_from_source_emission():
    state = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    operator = phase_operator(
        hazard=0.0,
        clear=0.0,
        encounter=0.0,
        release=0.0,
        escape=math.log(2.0),
        retained_recovery=0.0,
        dt=1.0,
    )
    final = apply_cycles(operator, state, 1)
    assert final[MOBILE] == pytest.approx(2.5)
    assert final[LOCAL_SOURCE_SLIP] == pytest.approx(0.0)
    assert final[LOCAL_FORWARD_SLIP] == pytest.approx(2.5)


def test_crack_advance_convects_old_slip_and_resharpens_tip():
    blunting = BluntingParameters(blunting_length_m=2.0e-6)
    state = np.array([1.0, 2.0, 0.5, 100.0, 50.0])
    before = effective_radius(state, blunting)
    final = convect_local_plasticity(state, 2.0e-6, blunting)
    assert final[MOBILE] == state[MOBILE]
    assert final[RETAINED] == state[RETAINED]
    assert final[SOURCE_ACTIVITY] == state[SOURCE_ACTIVITY]
    assert final[LOCAL_SOURCE_SLIP] == pytest.approx(100.0 / math.e)
    assert final[LOCAL_FORWARD_SLIP] == pytest.approx(50.0 / math.e)
    assert TIP_RADIUS_M < effective_radius(final, blunting) < before


def test_blunting_coefficients_are_shared_physical_coordinates():
    state = np.array([0.0, 0.0, 1.0, 10.0, 20.0])
    source_only = BluntingParameters(
        source_slip_coefficient=2.0, forward_slip_coefficient=0.0
    )
    forward_only = BluntingParameters(
        source_slip_coefficient=0.0, forward_slip_coefficient=1.0
    )
    assert effective_radius(state, source_only) == pytest.approx(
        TIP_RADIUS_M + 20.0 * BURGERS_M
    )
    assert effective_radius(state, forward_only) == pytest.approx(
        TIP_RADIUS_M + 20.0 * BURGERS_M
    )


def test_resolved_increment_conservatively_transfers_active_population_to_wake():
    blunting = BluntingParameters(blunting_length_m=4.0e-6)
    state = np.array([12.0, 8.0, 1.0, 100.0, 50.0])
    final, receipt = resolve_increment_advection(
        state, 0.5e-6, blunting, maximum_step_m=0.05e-6
    )
    assert final[MOBILE] < state[MOBILE]
    assert final[RETAINED] < state[RETAINED]
    assert final[WAKE_MOBILE] > 0.0
    assert final[WAKE_RETAINED] > 0.0
    assert receipt["active_to_wake_mobile"] > 0.0
    assert receipt["active_to_wake_retained"] > 0.0
    assert receipt["population_balance_residual"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < wake_tip_projection(blunting) < 1.0


def test_resolved_transaction_composes_and_can_outrun_short_wake():
    state = np.array([20.0, 10.0, 1.0, 30.0, 20.0])
    short = BluntingParameters(blunting_length_m=0.25e-6)
    long = BluntingParameters(blunting_length_m=10.0e-6)
    short_final, _ = resolve_increment_advection(
        state, 5.0e-6, short, maximum_step_m=0.05e-6
    )
    long_final, _ = resolve_increment_advection(
        state, 5.0e-6, long, maximum_step_m=0.05e-6
    )
    short_population = sum(short_final[i] for i in (MOBILE, RETAINED, WAKE_MOBILE, WAKE_RETAINED))
    long_population = sum(long_final[i] for i in (MOBILE, RETAINED, WAKE_MOBILE, WAKE_RETAINED))
    assert short_population < 0.1 * long_population
    whole, _ = resolve_increment_advection(
        state, 1.0e-6, long, maximum_step_m=0.05e-6
    )
    half, _ = resolve_increment_advection(
        state, 0.5e-6, long, maximum_step_m=0.05e-6
    )
    halves, _ = resolve_increment_advection(
        half, 0.5e-6, long, maximum_step_m=0.05e-6
    )
    assert halves == pytest.approx(whole, rel=1e-12, abs=1e-12)
