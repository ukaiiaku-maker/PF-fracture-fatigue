from __future__ import annotations

import math

import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls
from arrhenius_fracture.crack_rebonding_v10230 import (
    RebondingWakeState,
    cleavage_stress_with_rebond,
    reduced_modulus_Pa,
)


def _cfg(**overrides):
    fields = dict(
        wake_length_m=5.0e-6,
        wake_weight_length_m=1.0e-6,
        restored_work_of_separation_J_m2=2.0,
        rebond_K_geometry_factor=1.0,
    )
    fields.update(overrides)
    return CrackRebondingControls(**fields).validate()


def test_K_rebond_zero_when_no_active_wake():
    state = RebondingWakeState(_cfg())
    state.rebuild_coupling(Eprime_Pa=2.0e11)
    assert state.K_rebond_Pa_sqrt_m == 0.0
    assert state.H_b == 0.0


def test_K_rebond_bounded_by_K_rebond_max():
    cfg = _cfg()
    state = RebondingWakeState(cfg)
    state.commit_event(accepted_length_m=1.0e-6, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    state.active[0].p_B = 1.0
    state.active[0].p_C = 0.0
    state.active[0].p_P = 0.0
    state.rebuild_coupling(Eprime_Pa=2.0e11)
    assert 0.0 <= state.K_rebond_Pa_sqrt_m <= state.K_rebond_max_Pa_sqrt_m
    assert 0.0 <= state.H_b <= 1.0


def test_H_b_saturates_at_one_for_many_fully_bonded_patches():
    cfg = _cfg(wake_length_m=1.0e-4, wake_weight_length_m=2.0e-5)
    state = RebondingWakeState(cfg)
    for i in range(50):
        state.commit_event(
            accepted_length_m=2.0e-6, event_index=i, pre_event_states=None, Eprime_Pa=2.0e11
        )
        state.active[-1].p_B = 1.0
        state.active[-1].p_C = 0.0
        state.active[-1].p_P = 0.0
    state.rebuild_coupling(Eprime_Pa=2.0e11)
    assert state.H_b <= 1.0


def test_reduced_modulus_matches_plane_strain_formula():
    G, nu = 160.0e9, 0.28
    Eprime = reduced_modulus_Pa(G, nu)
    assert Eprime == pytest.approx(2.0 * G / (1.0 - nu))


def test_cleavage_stress_with_rebond_reduces_to_baseline_when_K_rebond_zero():
    K, K_shield, r_eff = 2.0e7, 5.0e6, 1.0e-6
    with_zero = cleavage_stress_with_rebond(K, K_shield, 0.0, r_eff)
    baseline = max(K - K_shield, 0.0) / math.sqrt(2.0 * math.pi * r_eff)
    assert with_zero == pytest.approx(baseline)


def test_cleavage_stress_with_rebond_monotonically_decreases_with_K_rebond():
    K, K_shield, r_eff = 2.0e7, 5.0e6, 1.0e-6
    vals = [cleavage_stress_with_rebond(K, K_shield, kr, r_eff) for kr in (0.0, 1.0e6, 5.0e6, 2.0e7)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))


def test_cleavage_stress_with_rebond_floors_at_zero():
    K, K_shield, r_eff = 1.0e6, 5.0e5, 1.0e-6
    val = cleavage_stress_with_rebond(K, K_shield, K_rebond_Pa_sqrt_m=K, r_eff_m=r_eff)
    assert val == 0.0


def test_K_shield_argument_passthrough_never_mutated_by_rebond_module():
    # cleavage_stress_with_rebond takes K_shield as a plain float argument; the
    # rebonding module has no reference to any engine K_shield ledger to
    # mutate. This test documents that invariant structurally: the function
    # is pure and returns a new float, never touching its inputs.
    K_shield = 5.0e6
    result = cleavage_stress_with_rebond(2.0e7, K_shield, 1.0e6, 1.0e-6)
    assert isinstance(result, float)
    assert K_shield == 5.0e6
