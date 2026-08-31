from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    CrackRebondingControls,
    advance_markov,
    bond_formation_rate,
    bond_rupture_rate,
    build_Q,
    cooperative_hazard,
)


def test_Q_columns_sum_to_zero():
    Q = build_Q(k_CB=0.3, k_BC=0.2, k_PC=0.1, k_CP=0.05)
    assert np.allclose(Q.sum(axis=0), 0.0)


@pytest.mark.parametrize("dt", [0.0, 1.0e-9, 1.0e-6, 1.0, 1.0e3])
def test_advance_markov_conserves_and_is_positive(dt):
    Q = build_Q(k_CB=1.5, k_BC=0.7, k_PC=0.4, k_CP=0.2)
    p0 = np.array([0.5, 0.3, 0.2])
    p1 = advance_markov(p0, Q, dt)
    assert p1.sum() == pytest.approx(1.0, abs=1.0e-10)
    assert (p1 >= -1.0e-12).all()


def test_advance_markov_substep_partition_invariance_constant_rate():
    Q = build_Q(k_CB=2.0, k_BC=0.5, k_PC=0.3, k_CP=0.1)
    p0 = np.array([0.6, 0.3, 0.1])
    dt = 3.7e-4
    one_shot = advance_markov(p0, Q, dt)
    n = 17
    stepped = p0.copy()
    for _ in range(n):
        stepped = advance_markov(stepped, Q, dt / n)
    assert np.allclose(one_shot, stepped, atol=1.0e-10)


def test_cooperative_hazard_m1_recovers_elementary_rate_exactly():
    lam_raw = 12345.6
    assert cooperative_hazard(lam_raw, m_h=1.0, tau_h=1.0e-6) == pytest.approx(lam_raw)


def test_cooperative_hazard_zero_raw_rate_is_zero():
    assert cooperative_hazard(0.0, m_h=3.0, tau_h=1.0e-6) == 0.0


def test_cooperative_hazard_small_x_limit_for_m_gt_1():
    # For m_h>1 and x = lambda_raw*tau_h << 1, gammainc(m,x)/tau ~ x^m/(m! tau) -> 0
    # much faster than the elementary rate itself, confirming distinctness from m_h=1.
    lam_raw = 1.0
    tau_h = 1.0e-9  # x = 1e-9, deeply sub-elementary regime
    k = cooperative_hazard(lam_raw, m_h=3.0, tau_h=tau_h)
    assert k < lam_raw
    assert k >= 0.0


def test_cooperative_hazard_monotonic_in_pressure_via_bond_formation_rate():
    cfg = CrackRebondingControls(
        bond_activation_volume_m3=2.0e-29,
        bond_barrier_eV=0.4,
        healing_cooperative_order=1.0,
    ).validate()
    rates = []
    for sigma in (0.0, 1.0e8, 5.0e8, 1.0e9):
        rate, _ = bond_formation_rate(sigma, T_K=300.0, chemistry_factor=1.0, cfg=cfg)
        rates.append(rate)
    assert all(b >= a for a, b in zip(rates, rates[1:]))
    assert rates[-1] > rates[0]


def test_bond_rupture_monotonic_in_opening_stress():
    cfg = CrackRebondingControls(
        rupture_activation_volume_m3=2.0e-29,
        rupture_barrier_eV=0.4,
    ).validate()
    rates = []
    for sigma in (0.0, 1.0e8, 5.0e8, 1.0e9):
        rate, _ = bond_rupture_rate(sigma, T_K=300.0, cfg=cfg, compressive_phase=False)
        rates.append(rate)
    assert all(b >= a for a, b in zip(rates, rates[1:]))
    assert rates[-1] > rates[0]


def test_bond_rupture_exactly_zero_during_compression():
    cfg = CrackRebondingControls(rupture_activation_volume_m3=2.0e-29).validate()
    rate, _ = bond_rupture_rate(1.0e9, T_K=300.0, cfg=cfg, compressive_phase=True)
    assert rate == 0.0


def test_chemistry_factor_suppresses_formation_monotonically():
    cfg = CrackRebondingControls(bond_activation_volume_m3=1.0e-29, bond_barrier_eV=0.3).validate()
    rates = []
    for chi in (1.0, 0.5, 0.1, 0.0):
        rate, _ = bond_formation_rate(5.0e8, T_K=300.0, chemistry_factor=chi, cfg=cfg)
        rates.append(rate)
    assert all(b <= a for a, b in zip(rates, rates[1:]))
    assert rates[-1] == 0.0


def test_chemistry_factor_zero_never_evaluates_log():
    cfg = CrackRebondingControls().validate()
    # Should not raise even with an extreme sigma that would otherwise blow up the exponent.
    rate, diag = bond_formation_rate(1.0e30, T_K=300.0, chemistry_factor=0.0, cfg=cfg)
    assert rate == 0.0
    assert diag["chemistry_zero"] is True


def test_barrier_floor_prevents_overflow():
    cfg = CrackRebondingControls(
        bond_activation_volume_m3=1.0e-25,  # huge, would drive barrier deeply negative
        bond_barrier_eV=0.5,
        bond_barrier_floor_eV=0.1,
    ).validate()
    rate, diag = bond_formation_rate(1.0e12, T_K=300.0, chemistry_factor=1.0, cfg=cfg)
    assert math.isfinite(rate)
    assert diag["floored"] is True
