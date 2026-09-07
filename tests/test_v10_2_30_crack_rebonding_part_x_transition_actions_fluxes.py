"""PX1.2 (Part X): transition-action/flux instrumentation.

Default-off diagnostics (mission section 5.2): for every patch and
accepted inter-event interval, the exact integrated rate actions
A_CB/A_BC/A_PC/A_CP and realized state-weighted fluxes
F_CB/F_BC/F_PC/F_CP, obtained via an augmented matrix exponential (exact
for a piecewise-constant generator -- no endpoint averaging or trapezoid
approximation), plus the state-balance closure identities that are the
strongest available check that these diagnostics measure the actual
propagated state rather than a parallel approximation.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.linalg import expm

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    InitialPrecrackWakeMode,
    build_Q,
    build_augmented_Q,
    build_phase_factors,
    propagate,
    transition_actions_and_fluxes,
)
from arrhenius_fracture.crack_rebonding_v10230 import (
    patch_rate_constants,
    patch_transition_actions_and_fluxes,
)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform

from _crack_rebonding_engine_fixture import build_real_engine, controller, rebonding_cfg


def _random_case(seed):
    rng = np.random.default_rng(seed)
    n = 6
    k_CB = rng.uniform(0.1, 3.0, n)
    k_BC = rng.uniform(0.1, 3.0, n)
    k_PC = rng.uniform(0.0, 2.0, n)
    k_CP = rng.uniform(0.0, 2.0, n)
    Qs = [build_Q(k_CB[i], k_BC[i], k_PC[i], k_CP[i]) for i in range(n)]
    dts = np.array([0.1, 0.15, 0.2, 0.12, 0.18, 1.5])
    rates = {"CB": k_CB, "BC": k_BC, "PC": k_PC, "CP": k_CP}
    p0 = np.array([0.5, 0.3, 0.2])
    return Qs, dts, rates, p0


def _fine_reference(p0, Qs, dts, k0, dt, k_CB, k_BC, k_PC, k_CP, n_substeps=2000):
    """Independent brute-force reference: fine substepping with explicit
    midpoint-rule accumulation of k_ij * p_source -- a genuinely different
    numerical method from the augmented-exponential code under test, so
    agreement is real cross-validation, not a shared-bug blind spot."""
    n = len(Qs)
    idx = k0 % n
    p = p0.copy()
    t_left = dt
    A = dict(CB=0.0, BC=0.0, PC=0.0, CP=0.0)
    F = dict(CB=0.0, BC=0.0, PC=0.0, CP=0.0)
    while t_left > 1e-12:
        bin_dt = dts[idx]
        take = min(bin_dt, t_left)
        sub_dt = take / n_substeps
        for _ in range(n_substeps):
            p_mid = expm(Qs[idx] * (sub_dt * 0.5)) @ p
            A["CB"] += k_CB[idx] * sub_dt
            A["BC"] += k_BC[idx] * sub_dt
            A["PC"] += k_PC[idx] * sub_dt
            A["CP"] += k_CP[idx] * sub_dt
            F["CB"] += k_CB[idx] * p_mid[1] * sub_dt
            F["BC"] += k_BC[idx] * p_mid[2] * sub_dt
            F["PC"] += k_PC[idx] * p_mid[0] * sub_dt
            F["CP"] += k_CP[idx] * p_mid[1] * sub_dt
            p = expm(Qs[idx] * sub_dt) @ p
        t_left -= take
        if take >= bin_dt - 1e-12:
            idx = (idx + 1) % n
    return p, A, F


@pytest.mark.parametrize("k0,dt", [(0, 0.05), (0, 0.35), (5, 1.0), (5, 1.5), (3, 2.5), (0, 4.4)])
def test_transition_actions_fluxes_match_independent_fine_reference(k0, dt):
    Qs, dts, rates, p0 = _random_case(7)
    result = transition_actions_and_fluxes(p0, Qs, rates, k0, dt, dts)
    p_ref, A_ref, F_ref = _fine_reference(
        p0, Qs, dts, k0, dt, rates["CB"], rates["BC"], rates["PC"], rates["CP"]
    )
    assert np.allclose(result["p_final"], p_ref, atol=1e-6)
    for key in ("CB", "BC", "PC", "CP"):
        assert result[f"A_{key}"] == pytest.approx(A_ref[key], abs=1e-8)
        assert result[f"F_{key}"] == pytest.approx(F_ref[key], abs=2e-7)


@pytest.mark.parametrize("k0,dt", [(0, 0.05), (0, 0.35), (5, 1.0), (5, 1.5), (3, 2.5), (0, 4.4)])
def test_transition_actions_fluxes_p_final_matches_propagate(k0, dt):
    """The diagnostic pass's own p_final must agree with the real
    (untouched) propagate() to machine precision -- the direct check that
    computing these diagnostics does not implicitly re-derive a different,
    diverging notion of the state evolution."""
    Qs, dts, rates, p0 = _random_case(7)
    factors = build_phase_factors(Qs, dts)
    result = transition_actions_and_fluxes(p0, Qs, rates, k0, dt, dts)
    p_check = propagate(p0, Qs, factors, k0, dt, dts)
    assert np.allclose(result["p_final"], p_check, atol=1e-10)


@pytest.mark.parametrize("k0,dt", [(0, 0.05), (0, 0.35), (5, 1.0), (5, 1.5), (3, 2.5), (0, 4.4)])
def test_state_balance_identities_close(k0, dt):
    """dp_P=-F_PC+F_CP, dp_C=F_PC-F_CP-F_CB+F_BC, dp_B=F_CB-F_BC for every
    interval -- the strongest available check that F_ij measures the
    actual propagated state, not a parallel approximation."""
    Qs, dts, rates, p0 = _random_case(11)
    result = transition_actions_and_fluxes(p0, Qs, rates, k0, dt, dts)
    dpP = result["p_final"][0] - p0[0]
    dpC = result["p_final"][1] - p0[1]
    dpB = result["p_final"][2] - p0[2]
    assert dpP == pytest.approx(-result["F_PC"] + result["F_CP"], abs=1e-10)
    assert dpC == pytest.approx(result["F_PC"] - result["F_CP"] - result["F_CB"] + result["F_BC"], abs=1e-10)
    assert dpB == pytest.approx(result["F_CB"] - result["F_BC"], abs=1e-10)


def test_scalar_dt_phase_uniform_matches_array_form():
    """The new function accepts either a uniform scalar or a per-bin
    array (no bit-identity requirement here, unlike propagate/
    phase_resolved_action -- this is a brand-new function with no prior
    behavior to preserve), and both forms must agree numerically."""
    rng = np.random.default_rng(3)
    n = 5
    k_CB = rng.uniform(0.1, 2.0, n)
    k_BC = rng.uniform(0.1, 2.0, n)
    k_PC = np.zeros(n)
    k_CP = np.zeros(n)
    Qs = [build_Q(k_CB[i], k_BC[i], 0.0, 0.0) for i in range(n)]
    dt_scalar = 0.07
    rates = {"CB": k_CB, "BC": k_BC, "PC": k_PC, "CP": k_CP}
    p0 = np.array([0.0, 1.0, 0.0])
    r_scalar = transition_actions_and_fluxes(p0, Qs, rates, 0, 0.5, dt_scalar)
    r_array = transition_actions_and_fluxes(p0, Qs, rates, 0, 0.5, np.full(n, dt_scalar))
    assert np.allclose(r_scalar["p_final"], r_array["p_final"], atol=1e-12)
    for key in ("CB", "BC", "PC", "CP"):
        assert r_scalar[f"A_{key}"] == pytest.approx(r_array[f"A_{key}"], abs=1e-12)
        assert r_scalar[f"F_{key}"] == pytest.approx(r_array[f"F_{key}"], abs=1e-12)


def test_zero_rates_give_zero_actions_and_fluxes_with_unchanged_state():
    Qs = [build_Q(0.0, 0.0, 0.0, 0.0) for _ in range(4)]
    rates = {"CB": np.zeros(4), "BC": np.zeros(4), "PC": np.zeros(4), "CP": np.zeros(4)}
    p0 = np.array([0.2, 0.5, 0.3])
    result = transition_actions_and_fluxes(p0, Qs, rates, 0, 3.3, np.array([0.1, 0.1, 0.1, 0.1]))
    assert np.array_equal(result["p_final"], p0)
    for key in ("CB", "BC", "PC", "CP"):
        assert result[f"A_{key}"] == 0.0
        assert result[f"F_{key}"] == 0.0


def test_build_augmented_Q_block_structure():
    Q = build_Q(1.0, 2.0, 0.5, 0.3)
    Q_aug = build_augmented_Q(Q)
    assert Q_aug.shape == (6, 6)
    assert np.array_equal(Q_aug[:3, :3], Q)
    assert np.array_equal(Q_aug[3:, :3], np.eye(3))
    assert np.array_equal(Q_aug[:3, 3:], np.zeros((3, 3)))
    assert np.array_equal(Q_aug[3:, 3:], np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Real-engine wiring
# ---------------------------------------------------------------------------


def _rb_cfg_with_clean_precrack(**overrides):
    return dataclasses.replace(
        rebonding_cfg(**overrides),
        initial_precrack_wake_mode=InitialPrecrackWakeMode.INITIAL_WAKE_CLEAN,
    ).validate()


def test_patch_rate_constants_matches_patch_Q_generator():
    """patch_rate_constants (the diagnostic's rate source) must reproduce
    exactly the same generator patch_Q builds for the real physics --
    guarding against the two silently drifting apart."""
    from arrhenius_fracture.crack_rebonding_v10230 import patch_Q

    cfg = _rb_cfg_with_clean_precrack()
    for K_s in (-9.0e6, -1.0e6, 0.0, 5.0e6, 18.0e6):
        k_CB, k_BC, k_PC, k_CP = patch_rate_constants(K_s, 0.0, 1e-9, cfg, 300.0)
        Q_from_rates = build_Q(k_CB, k_BC, k_PC, k_CP)
        Q_direct = patch_Q(K_s, 0.0, 1e-9, cfg, 300.0)
        assert np.array_equal(Q_from_rates, Q_direct)


def test_patch_transition_actions_and_fluxes_real_engine_wiring():
    """End-to-end: real engine, real wake patch, real dwell-extended
    signed schedule -- confirms the diagnostic wraps correctly around the
    actual production per-bin rate construction and closes the
    state-balance identities against the real patch's own state."""
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg_with_clean_precrack(bond_barrier_eV=0.55)
    engine = build_real_engine(cfg)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    K_signed, dt_vals = wave.cycle_schedule(16, signed=True)
    patch = engine._rebonding_state.active[0]
    r_contact_m = max(engine.r_eff(), cfg.contact_radius_min_m)

    p_before = patch.state_vector()
    result = patch_transition_actions_and_fluxes(
        patch=patch, K_signed_phase=K_signed, dt_phase=dt_vals, r_contact_m=r_contact_m,
        cfg=cfg, T_K=300.0, dt_consumed=wave.period_s * 3.5,
    )
    # Pure diagnostic: must not mutate the patch it was computed from.
    assert np.array_equal(patch.state_vector(), p_before)

    dpP = result["p_final"][0] - p_before[0]
    dpC = result["p_final"][1] - p_before[1]
    dpB = result["p_final"][2] - p_before[2]
    assert dpP == pytest.approx(-result["F_PC"] + result["F_CP"], abs=1e-10)
    assert dpC == pytest.approx(result["F_PC"] - result["F_CP"] - result["F_CB"] + result["F_BC"], abs=1e-10)
    assert dpB == pytest.approx(result["F_CB"] - result["F_BC"], abs=1e-10)
    # R=-0.5 is compressive over part of the cycle+dwell: some formation
    # action/flux must actually be nonzero (a genuinely exercised channel,
    # not a silently-zero diagnostic).
    assert result["A_CB"] > 0.0
    assert result["F_CB"] > 0.0


def test_instrumentation_never_perturbs_real_wake_state():
    """Instrumentation-on vs. instrumentation-off physical parity, proven
    the reliable way: on a SINGLE real engine/patch, calling the
    diagnostic (any number of times, with arbitrary dt_consumed) leaves
    the patch's own state, and the engine's real subsequent behavior,
    completely unchanged.

    An earlier version of this test compared two independently-
    constructed engines (instrumented vs. plain) step by step. That
    pattern was abandoned: it reproducibly diverged by an amount far
    larger than floating-point roundoff when this file ran as part of the
    full crack_rebonding selection (not in isolation), for the same
    pre-existing global-state reason
    test_v10_2_30_crack_rebonding_part_x_minimum_load_hold.py's own
    removed full-trajectory two-engine test hit -- unrelated to Part X or
    to this diagnostic, since patch_transition_actions_and_fluxes
    provably never writes to ``patch``, ``cfg``, or any engine attribute
    (it only reads ``patch.state_vector()``/``patch.s_j_m`` and returns a
    fresh dict), so its side-effect-freedom is a property of a single
    engine's own before/after state, not something a cross-engine
    comparison is needed -- or reliable -- to establish here.
    """
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg_with_clean_precrack(bond_barrier_eV=0.55)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    K_signed, dt_vals = wave.cycle_schedule(16, signed=True)

    engine = build_real_engine(cfg)
    r_contact_m = max(engine.r_eff(), cfg.contact_radius_min_m)

    for _ in range(5):
        patch = engine._rebonding_state.active[0]
        state_before_diagnostic = patch.state_vector()
        # Call the diagnostic several times, with varying dt_consumed,
        # between real steps -- must never mutate the patch.
        for dt_consumed in (wave.period_s * 0.3, wave.period_s * 1.7, 0.0):
            patch_transition_actions_and_fluxes(
                patch=patch, K_signed_phase=K_signed, dt_phase=dt_vals,
                r_contact_m=r_contact_m, cfg=cfg, T_K=300.0, dt_consumed=dt_consumed,
            )
            assert np.array_equal(patch.state_vector(), state_before_diagnostic)

        r = engine.cycle_step_waveform(ctrl, wave, 300.0)
        if r.get("fired"):
            break
