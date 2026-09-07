"""PX1.3 (Part X): RB3 (PASSIVATION_GATED_REBOND) audit.

patch_Q already implements the full P<->C<->B generator for
RebondModelLevel.PASSIVATION_GATED_REBOND (depassivation_rate/
repassivation_rate wired in crack_rebonding_v10230.py). This is an
audit/test-coverage pass on existing code through the real A_NATIVE
production engine, not new physics (mission section 5.3): live,
nonzero, correctly-signed P->C depassivation and C->P repassivation;
exact probability conservation/nonnegativity; compression-gated
depassivation; exact zero contact formation at K>=0; repassivation only
at its qualified (tensile) phase condition; fresh-surface P/C
initialization; rollback/checkpoint restoration of all three states; no
direct energy-gate/emission contamination.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel,
    CrackRebondingControls,
    InitialPrecrackWakeMode,
    RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_v10230 import (
    WakePatch,
    install_crack_rebonding,
    patch_Q,
    patch_rate_constants,
    patch_transition_actions_and_fluxes,
    restore_rebonding_checkpoint,
    serialize_rebonding_checkpoint,
)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.persistent_site_coupled_hazard_v10229 import _commit_constant_segment

from _crack_rebonding_engine_fixture import build_real_engine, controller


def _rb3_cfg(**overrides):
    base = dict(
        enabled=True,
        model_level=RebondModelLevel.PASSIVATION_GATED_REBOND,
        contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY,
        wake_length_m=5.0e-4,
        wake_weight_length_m=5.0e-7,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=0.5,
        bond_activation_volume_m3=0.0,
        bond_barrier_eV=0.5,
        bond_attempt_frequency_s=1.0e10,
        rupture_activation_volume_m3=0.0,
        rupture_barrier_eV=1.0,
        rupture_attempt_frequency_s=1.0e10,
        depassivation_activation_volume_m3=0.0,
        depassivation_barrier_eV=0.5,
        depassivation_attempt_frequency_s=1.0e10,
        repassivation_barrier_eV=0.5,
        repassivation_attempt_frequency_s=1.0e10,
        chemistry_factor=1.0,
        initial_precrack_wake_mode=InitialPrecrackWakeMode.INITIAL_WAKE_PASSIVATED,
    )
    base.update(overrides)
    return CrackRebondingControls(**base).validate()


# ---------------------------------------------------------------------------
# 1. Live, nonzero, correctly-signed transitions
# ---------------------------------------------------------------------------


def test_depassivation_active_only_in_compressive_phase():
    """P->C (depassivation) must be exactly zero at K>=0 and strictly
    positive at K<0 (compressive)."""
    cfg = _rb3_cfg()
    k_PC_tension, _, _, _ = patch_rate_constants(5.0e6, 0.0, 1e-9, cfg, 300.0)
    k_PC_compression, _, _, _ = patch_rate_constants(-5.0e6, 0.0, 1e-9, cfg, 300.0)
    assert k_PC_tension == 0.0
    assert k_PC_compression > 0.0


def test_repassivation_active_only_in_tensile_phase():
    """C->P (repassivation) must be exactly zero at K<=0 and strictly
    positive at K>0 (tensile) -- gated on K_s_sign, not on activation
    volume/stress at all (repassivation_rate takes no stress argument)."""
    cfg = _rb3_cfg()
    _, _, _, k_CP_compression = patch_rate_constants(-5.0e6, 0.0, 1e-9, cfg, 300.0)
    _, _, _, k_CP_zero = patch_rate_constants(0.0, 0.0, 1e-9, cfg, 300.0)
    _, _, _, k_CP_tension = patch_rate_constants(5.0e6, 0.0, 1e-9, cfg, 300.0)
    assert k_CP_compression == 0.0
    assert k_CP_zero == 0.0
    assert k_CP_tension > 0.0


def test_formation_and_rupture_active_as_in_rb2():
    """C->B (formation, compressive) and B->C (rupture, tensile) behave
    identically to the already-qualified RB2 CLEAN_REVERSIBLE_REBOND
    contract -- RB3 only adds the P<->C channel on top."""
    cfg = _rb3_cfg()
    k_CB_comp, k_BC_comp, _, _ = patch_rate_constants(-5.0e6, 0.0, 1e-9, cfg, 300.0)
    k_CB_tens, k_BC_tens, _, _ = patch_rate_constants(5.0e6, 0.0, 1e-9, cfg, 300.0)
    assert k_CB_comp > 0.0 and k_BC_comp == 0.0
    assert k_CB_tens == 0.0 and k_BC_tens > 0.0


def test_all_four_transitions_live_through_real_engine_trajectory():
    """External review, correctly: the original version of this test only
    asserted p_P<1 and (p_C>0 or p_B>0) -- consistent with as few as ONE
    of the four transitions ever firing. Strengthened to require ALL
    eight quantities (A_PC, A_CP, A_CB, A_BC, F_PC, F_CP, F_CB, F_BC)
    strictly positive, measured directly via PX1.2's exact
    patch_transition_actions_and_fluxes over the precise per-segment
    dt_consumed each committed segment actually used (not the requested
    block duration), summed across the whole trajectory, with the
    cumulative state-balance identities closing against the real
    observed state change."""
    ctrl = controller(n_phase=16)
    cfg = _rb3_cfg()
    engine = build_real_engine(cfg)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0)
    r_contact_m = max(engine.r_eff(), cfg.contact_radius_min_m)
    n_phase = 16
    K_signed, dt_signed = wave.cycle_schedule(n_phase, signed=True)

    patch = engine._rebonding_state.active[0]
    p_initial = patch.state_vector().copy()
    totals = {"A_PC": 0.0, "A_CP": 0.0, "A_CB": 0.0, "A_BC": 0.0,
              "F_PC": 0.0, "F_CP": 0.0, "F_CB": 0.0, "F_BC": 0.0}

    for _ in range(30):
        state_before = patch.state_vector().copy()
        result = _commit_constant_segment(
            engine, ctrl, wave, 300.0, 1.0, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3,
        )
        dt_consumed = float(result.get("dt_consumed", wave.period_s))
        # Independently measure this segment's exact diagnostics from the
        # pre-segment state over the exact consumed duration (not the
        # requested block), then verify against the real observed
        # post-segment state before folding into the running totals.
        patch_probe = WakePatch(
            patch_id=-1, length_m=patch.length_m, s_j_m=patch.s_j_m,
            p_P=state_before[0], p_C=state_before[1], p_B=state_before[2],
            creation_event_index=-1,
        )
        diag = patch_transition_actions_and_fluxes(
            patch=patch_probe, K_signed_phase=K_signed, dt_phase=dt_signed,
            r_contact_m=r_contact_m, cfg=cfg, T_K=300.0, dt_consumed=dt_consumed,
        )
        assert np.allclose(diag["p_final"], patch.state_vector(), atol=1e-8)
        for key in totals:
            totals[key] += diag[key]

    for key, value in totals.items():
        assert value > 0.0, f"{key} was not strictly positive over the trajectory ({value})"

    p_final = patch.state_vector()
    dpP, dpC, dpB = p_final - p_initial
    assert dpP == pytest.approx(-totals["F_PC"] + totals["F_CP"], abs=1e-6)
    assert dpC == pytest.approx(totals["F_PC"] - totals["F_CP"] - totals["F_CB"] + totals["F_BC"], abs=1e-6)
    assert dpB == pytest.approx(totals["F_CB"] - totals["F_BC"], abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Probability conservation and nonnegativity
# ---------------------------------------------------------------------------


def test_probability_conservation_and_nonnegativity_through_trajectory():
    ctrl = controller(n_phase=16)
    cfg = _rb3_cfg()
    engine = build_real_engine(cfg)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0003)
    for _ in range(40):
        _commit_constant_segment(engine, ctrl, wave, 300.0, 1.0, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
        patch = engine._rebonding_state.active[0]
        total = patch.p_P + patch.p_C + patch.p_B
        assert total == pytest.approx(1.0, abs=1e-8)
        assert patch.p_P >= -1e-12
        assert patch.p_C >= -1e-12
        assert patch.p_B >= -1e-12


# ---------------------------------------------------------------------------
# 3. Exact zero contact formation at K>=0 (RB3, not just RB2)
# ---------------------------------------------------------------------------


def test_positive_R_produces_zero_formation_and_zero_depassivation():
    ctrl = controller(n_phase=16)
    cfg = _rb3_cfg()
    engine = build_real_engine(cfg)
    wave = FatigueWaveform(Kmax=18.0e6, R=0.1, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    _commit_constant_segment(engine, ctrl, wave, 300.0, 30.0, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
    patch = engine._rebonding_state.active[0]
    # Started INITIAL_WAKE_PASSIVATED (p_P=1); with K>=0 throughout (no
    # compression anywhere in the schedule, including the dwell), neither
    # depassivation (P->C) nor formation (C->B, requires C first) can
    # occur, but repassivation-from-nothing is a no-op on p_P=1 -- state
    # must remain exactly at its initial passivated value.
    assert patch.p_P == 1.0
    assert patch.p_C == 0.0
    assert patch.p_B == 0.0


# ---------------------------------------------------------------------------
# 4. Fresh-surface P/C initialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clean_fraction", [0.0, 0.3, 1.0])
def test_fresh_surface_clean_fraction_sets_new_patch_split(clean_fraction):
    """A newly created patch (commit_event) must split p_C/p_P exactly
    according to fresh_surface_clean_fraction, independent of the
    passivation model level -- this governs patch creation, not the
    per-instant generator."""
    cfg = _rb3_cfg(fresh_surface_clean_fraction=clean_fraction)
    from arrhenius_fracture.crack_rebonding_v10230 import RebondingWakeState, reduced_modulus_Pa

    state = RebondingWakeState(cfg)
    state.commit_event(
        accepted_length_m=1.0e-6, event_index=0, pre_event_states=None,
        Eprime_Pa=reduced_modulus_Pa(80.0e9, 0.3),
    )
    new_patch = state.active[-1]
    assert new_patch.p_C == pytest.approx(clean_fraction)
    assert new_patch.p_P == pytest.approx(1.0 - clean_fraction)
    assert new_patch.p_B == 0.0


# ---------------------------------------------------------------------------
# 5. Rollback / checkpoint restoration of all three states
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_all_three_states_exactly():
    cfg = _rb3_cfg()
    engine = build_real_engine(cfg)
    ctrl = controller(n_phase=16)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0)
    for _ in range(10):
        _commit_constant_segment(engine, ctrl, wave, 300.0, 1.0, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
    patch_before = engine._rebonding_state.active[0]
    p_before = (patch_before.p_P, patch_before.p_C, patch_before.p_B)

    payload = serialize_rebonding_checkpoint(engine)
    assert payload is not None

    # Perturb the live state, then restore -- must recover exactly.
    patch_before.set_state(np.array([0.9, 0.05, 0.05]))
    assert (patch_before.p_P, patch_before.p_C, patch_before.p_B) != p_before

    restore_rebonding_checkpoint(engine, payload)
    patch_after = engine._rebonding_state.active[0]
    assert patch_after.p_P == pytest.approx(p_before[0])
    assert patch_after.p_C == pytest.approx(p_before[1])
    assert patch_after.p_B == pytest.approx(p_before[2])


def test_snapshot_restore_round_trip_via_wake_state_directly():
    """RebondingWakeState.snapshot()/restore() (used for transactional
    event rollback, distinct from the checkpoint serialization above)
    also round-trips all three per-patch states exactly."""
    cfg = _rb3_cfg()
    engine = build_real_engine(cfg)
    ctrl = controller(n_phase=16)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0)
    for _ in range(10):
        _commit_constant_segment(engine, ctrl, wave, 300.0, 1.0, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
    state = engine._rebonding_state
    snap = state.snapshot()
    p_before = state.active[0].state_vector().copy()

    state.active[0].set_state(np.array([0.1, 0.1, 0.8]))
    assert not np.allclose(state.active[0].state_vector(), p_before)

    state.restore(snap)
    assert np.allclose(state.active[0].state_vector(), p_before)


# ---------------------------------------------------------------------------
# 6. No direct energy-gate/emission contamination
# ---------------------------------------------------------------------------


def test_rb3_does_not_change_ordinary_emission_relative_to_rb0():
    """RB3 enters only sig_cleave (via cleavage_stress_with_rebond); the
    ordinary emission/plasticity channel (fed by K_values / sigma_tip, not
    sig_cleave) must be unaffected by whether rebonding is installed.

    Per external review: comparing two independently-constructed engines
    is unreliable in this suite (pre-existing class-level state can leak
    across test execution order -- see test_v10_2_30_crack_rebonding_
    part_x_engine_reproducibility.py for the root cause and fix). Uses a
    SINGLE engine's own before/after channel-level calculation instead:
    mu_emit is a deterministic function of engine.mpz/sigma_tip/material
    state via preview_cycle_waveform, with no hazard-RNG dependence at
    all, so calling it before and after installing RB3 on the identical
    engine is a strictly stronger and more direct test than comparing two
    separate engines could ever be.
    """
    ctrl = controller(n_phase=16)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0)
    engine = build_real_engine(None)

    pred_before = engine.preview_cycle_waveform(ctrl, wave, 300.0)
    install_crack_rebonding(engine, _rb3_cfg())
    pred_after = engine.preview_cycle_waveform(ctrl, wave, 300.0)

    assert pred_after.mu_emit == pytest.approx(pred_before.mu_emit, rel=1e-12)
    assert pred_after.avg_sigma_tip == pytest.approx(pred_before.avg_sigma_tip, rel=1e-12)
