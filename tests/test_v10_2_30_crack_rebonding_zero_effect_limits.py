from __future__ import annotations

import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls, RebondModelLevel
from arrhenius_fracture.crack_rebonding_v10230 import (
    RebondingWakeState,
    patch_Q,
)


def _active_cfg(**overrides):
    base = dict(
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        bond_activation_volume_m3=2.0e-29,
        bond_barrier_eV=0.3,
        rupture_activation_volume_m3=2.0e-29,
        rupture_barrier_eV=0.3,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=1.0,
        bond_attempt_frequency_s=1.0e13,
    )
    base.update(overrides)
    return CrackRebondingControls(**base).validate()


def test_chemistry_factor_zero_gives_zero_formation_hazard():
    cfg = _active_cfg(chemistry_factor=0.0)
    Q = patch_Q(-1.0e7, s_j_m=1.0e-7, r_contact_m=1.0e-8, cfg=cfg, T_K=300.0)
    # k_CB feeds row B (index 2) from column C (index 1); with
    # chemistry_factor=0 bond formation must be exactly zero regardless of
    # compressive stress.
    assert Q[2, 1] == 0.0


def test_restored_work_of_separation_zero_gives_zero_K_rebond_regardless_of_bonding():
    cfg = _active_cfg(restored_work_of_separation_J_m2=0.0)
    state = RebondingWakeState(cfg)
    state.commit_event(accepted_length_m=1.0e-6, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    state.active[0].p_B = 1.0
    state.active[0].p_C = 0.0
    state.active[0].p_P = 0.0
    state.rebuild_coupling(Eprime_Pa=2.0e11)
    assert state.K_rebond_Pa_sqrt_m == 0.0


def test_rebond_K_geometry_factor_zero_gives_zero_K_rebond():
    cfg = _active_cfg(rebond_K_geometry_factor=0.0)
    state = RebondingWakeState(cfg)
    state.commit_event(accepted_length_m=1.0e-6, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    state.active[0].p_B = 1.0
    state.rebuild_coupling(Eprime_Pa=2.0e11)
    assert state.K_rebond_Pa_sqrt_m == 0.0


def test_bond_attempt_frequency_zero_prevents_any_formation():
    cfg = _active_cfg(bond_attempt_frequency_s=0.0)
    Q = patch_Q(-1.0e7, s_j_m=1.0e-7, r_contact_m=1.0e-8, cfg=cfg, T_K=300.0)
    assert Q[2, 1] == 0.0
