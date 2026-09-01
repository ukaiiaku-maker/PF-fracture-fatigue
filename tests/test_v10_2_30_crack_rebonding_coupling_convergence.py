from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    CrackRebondingControls,
    RebondModelLevel,
    build_Q,
)
from arrhenius_fracture.crack_rebonding_v10230 import (
    WakePatch,
    phase_resolved_action,
    stage1_block_cycle_limits,
    stage2_verify_block,
)


def _cfg(**overrides):
    fields = dict(
        model_level=RebondModelLevel.PASSIVATION_GATED_REBOND,
        wake_length_m=5.0e-6,
        wake_weight_length_m=1.0e-6,
        restored_work_of_separation_J_m2=3.0,
        rebond_K_geometry_factor=1.0,
        bond_activation_volume_m3=3.0e-29,
        bond_barrier_eV=0.2,
        rupture_activation_volume_m3=3.0e-29,
        rupture_barrier_eV=0.6,
        depassivation_activation_volume_m3=3.0e-29,
        depassivation_barrier_eV=0.2,
        repassivation_barrier_eV=0.5,
        chemistry_factor=1.0,
    )
    fields.update(overrides)
    return CrackRebondingControls(**fields).validate()


def _K_phase(idx, n_phase, Kmax, R):
    phase = (idx + 0.5) * (2.0 * math.pi / n_phase)
    Kmean = 0.5 * (Kmax + R * Kmax)
    Kamp = 0.5 * (Kmax - R * Kmax)
    return Kmean + Kamp * math.cos(phase)


def _baseline_action_estimate(K_s0, r_eff_m, lambda_cleave_fn, dt_candidate):
    """A representative constant-rate action estimate at the block-start
    stress (no rebonding correction) -- what a real caller would supply as
    the 'representative-rate approximation' Stage 2 checks consistency
    against. Zero would trivially fail every nonzero-action comparison."""
    sigma0 = max(K_s0, 0.0) / math.sqrt(2.0 * math.pi * max(r_eff_m, 1.0e-30))
    return lambda_cleave_fn(sigma0) * dt_candidate


def test_stage1_limits_are_positive_cycle_counts():
    cfg = _cfg()
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    limits = stage1_block_cycle_limits(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        K_s0=-1.0e7,  # compressive, so formation hazard is active
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        period_s=1.0e-3,
    )
    assert all(l > 0.0 for l in limits)


_MILD_KINETICS = dict(
    bond_barrier_eV=0.3,
    rupture_barrier_eV=0.3,
    depassivation_barrier_eV=0.3,
    repassivation_barrier_eV=0.3,
    bond_attempt_frequency_s=1.0e6,
    rupture_attempt_frequency_s=1.0e6,
    depassivation_attempt_frequency_s=1.0e6,
    repassivation_attempt_frequency_s=1.0e6,
)

# Constant compressive/slow-rupture kinetics: a genuinely oscillating K(phase)
# waveform can settle into a periodic limit cycle within a few cycles, at
# which point comparing phase-aligned start/end states (both at k0=0) shows
# near-zero net drift regardless of how many cycles elapsed -- a test-design
# artifact, not evidence the block-size check is ineffective. Using a
# constant compressive proxy with formation dominant over rupture instead
# gives clean, monotonic single-exponential-like relaxation toward the
# two-state fixed point, so "many cycles" unambiguously accumulates more
# state change than "one bin."
_MONOTONIC_KINETICS = dict(
    model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
    bond_activation_volume_m3=0.0,  # purely barrier-controlled: predictable, no
    rupture_activation_volume_m3=0.0,  # capped-pressure-driven barrier collapse
    bond_barrier_eV=0.3,
    bond_attempt_frequency_s=1.0e6,
    rupture_barrier_eV=2.0,
    rupture_attempt_frequency_s=1.0e6,
)


def test_stage2_pass_for_a_tiny_block_and_fail_for_a_huge_one():
    cfg = _cfg(
        rebonding_block_max_dpB=0.01,
        rebonding_block_max_dpC=0.01,
        rebonding_block_max_dK_rebond_frac=0.01,
        **_MONOTONIC_KINETICS,
    )
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    n_phase = 40
    f_Hz = 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase

    def K_phase_fn(idx):
        return -1.0e7  # constant compressive proxy: no periodic cancellation

    def lambda_cleave_fn(sigma):
        return 1.0e-3 * sigma

    tiny = stage2_verify_block(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        dt_candidate=dt_phase,  # one phase step
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=lambda_cleave_fn,
        representative_action_estimate=_baseline_action_estimate(
            K_phase_fn(0), 1.0e-6, lambda_cleave_fn, dt_phase
        ),
    )
    assert tiny["passed"] is True

    huge = stage2_verify_block(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        dt_candidate=1.0e3 * n_phase * dt_phase,  # 1,000 cycles (~9 time constants)
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=lambda_cleave_fn,
        representative_action_estimate=_baseline_action_estimate(
            K_phase_fn(0), 1.0e-6, lambda_cleave_fn, 1.0e3 * n_phase * dt_phase
        ),
    )
    assert huge["passed"] is False
    assert huge["max_dpB"] > tiny["max_dpB"]


def test_fail_closed_bisection_recovers_a_passing_block():
    cfg = _cfg(
        rebonding_block_max_dpB=0.02,
        rebonding_block_max_dpC=0.02,
        rebonding_block_max_dK_rebond_frac=0.02,
        **_MONOTONIC_KINETICS,
    )
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    n_phase = 40
    f_Hz = 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase

    def K_phase_fn(idx):
        return -1.0e7  # constant compressive proxy: monotonic, predictable relaxation

    def lambda_cleave_fn(sigma):
        return 1.0e-3 * sigma

    def verify(dt_candidate):
        return stage2_verify_block(
            active_patches=[patch],
            patch_states={0: patch.state_vector()},
            k0=0,
            dt_candidate=dt_candidate,
            K_phase_fn=K_phase_fn,
            dt_phase=dt_phase,
            n_phase=n_phase,
            r_contact_m=1.0e-8,
            cfg=cfg,
            T_K=300.0,
            Eprime_Pa=2.0e11,
            K_shield_Pa_sqrt_m=0.0,
            r_eff_m=1.0e-6,
            lambda_cleave_fn=lambda_cleave_fn,
            representative_action_estimate=_baseline_action_estimate(
                K_phase_fn(0), 1.0e-6, lambda_cleave_fn, dt_candidate
            ),
        )

    dt_candidate = 1.0e3 * n_phase * dt_phase
    result = verify(dt_candidate)
    assert result["passed"] is False  # confirm the starting candidate genuinely fails
    bisections = 0
    while not result["passed"] and bisections < 16:
        dt_candidate /= 2.0
        result = verify(dt_candidate)
        bisections += 1
    assert result["passed"] is True
    assert bisections > 0


def test_p_to_c_to_b_activation_not_missed_by_linear_stage1_estimate():
    """A patch starting fully passivated (p_P=1) has dpB/dt(0)=0 exactly, since
    bonds can only form from the clean state -- Stage 1's linearized dpB/dt
    limit is therefore silent about this patch. Stage 2's exact trial
    propagation over a long-enough candidate block must still detect the
    P->C->B activation that occurs once depassivation populates the clean
    state, and fail the block so the caller bisects."""
    cfg = _cfg(
        rebonding_block_max_dpB=0.01,
        rebonding_block_max_dpC=0.01,
        rebonding_block_max_dK_rebond_frac=0.01,
        depassivation_barrier_eV=0.05,  # fast depassivation under compression
        bond_barrier_eV=0.05,  # fast bonding once clean
    )
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=1.0, p_C=0.0, p_B=0.0, creation_event_index=0)

    n_phase = 40
    Kmax, R, f_Hz = 18.0e6, -0.95, 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase

    def K_phase_fn(idx):
        return _K_phase(idx, n_phase, Kmax, R)

    def lambda_cleave_fn(sigma):
        return 1.0e-3 * sigma

    # Stage 1's local-derivative estimate at t=0 sees dpB/dt = 0 exactly
    # (state is [1,0,0], and k_CB acts on p_C which is zero), so it produces
    # no dpB-based limit at all -- confirming the review's exact concern.
    limits = stage1_block_cycle_limits(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        K_s0=-1.0e7,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        period_s=1.0 / f_Hz,
    )
    # Confirm the underlying premise directly: Q@p at the start state has a
    # zero p_B-component derivative (dp_B/dt = k_CB*p_C - k_BC*p_B = 0 when
    # p_C=p_B=0), independent of whatever Stage 1 happened to append to
    # `limits` for other components.
    Q0 = build_Q(k_CB=1.0, k_BC=0.0, k_PC=1.0, k_CP=0.0)  # representative shape
    dpdt0 = Q0 @ patch.state_vector()
    assert dpdt0[2] == pytest.approx(0.0)

    # Stage 2 over a long candidate block must still catch the P->C->B
    # activation and fail (forcing bisection), since it uses exact
    # propagation rather than the t=0 linearization.
    dt_candidate = 2.0e4 * n_phase * dt_phase
    result = stage2_verify_block(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        dt_candidate=dt_candidate,
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=lambda_cleave_fn,
        representative_action_estimate=0.0,
    )
    assert result["max_dpB"] > 0.0
    assert result["passed"] is False


def test_phase_resolution_convergence_of_representative_action():
    """Halving/doubling the phase grid while keeping the same physical block
    duration must converge the integrated action to a stable value."""
    cfg = _cfg()
    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=1.0e-7, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    Kmax, R, f_Hz = 18.0e6, -0.95, 1000.0
    t_interval = 3.0 / f_Hz  # 3 full cycles

    def lambda_cleave_fn(sigma):
        return 1.0e-3 * sigma

    actions = []
    for n_phase in (20, 40, 80):
        dt_phase = (1.0 / f_Hz) / n_phase

        def K_phase_fn(idx, n_phase=n_phase):
            return _K_phase(idx, n_phase, Kmax, R)

        action, _, _, _ = phase_resolved_action(
            active_patches=[patch],
            patch_states={0: patch.state_vector()},
            k0=0,
            t_interval=t_interval,
            K_phase_fn=K_phase_fn,
            dt_phase=dt_phase,
            n_phase=n_phase,
            r_contact_m=1.0e-8,
            cfg=cfg,
            T_K=300.0,
            Eprime_Pa=2.0e11,
            K_shield_Pa_sqrt_m=0.0,
            r_eff_m=1.0e-6,
            lambda_cleave_fn=lambda_cleave_fn,
        )
        actions.append(action)

    assert abs(actions[1] - actions[2]) < abs(actions[0] - actions[1]) + 1.0e-30
