from __future__ import annotations

import math

import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    REFERENCE_ACTION_PRESETS,
    CrackRebondingControls,
    freeze_reference_action_preset,
    solve_reference_action_barriers,
    two_state_fixed_point,
    two_state_iterate,
)


def test_two_state_fixed_point_matches_manual_iteration():
    A_on, A_off = 1.3, 0.7
    fp = two_state_fixed_point(A_on, A_off)
    history = two_state_iterate(0.0, A_on, A_off, n_cycles=400)
    assert history[-1] == pytest.approx(fp["b_star"], abs=1.0e-6)


def test_two_state_iteration_converges_from_any_start():
    A_on, A_off = 2.0, 0.3
    fp = two_state_fixed_point(A_on, A_off)
    for b0 in (0.0, 0.3, 0.9, 1.0):
        history = two_state_iterate(b0, A_on, A_off, n_cycles=500)
        assert history[-1] == pytest.approx(fp["b_star"], abs=1.0e-6)


def test_two_state_survival_probability_formula():
    A_on, A_off = 0.5, 0.2
    fp = two_state_fixed_point(A_on, A_off)
    expected = (1.0 - math.exp(-A_on)) * math.exp(-A_off)
    assert fp["P_survive"] == pytest.approx(expected)


def test_two_state_degenerate_zero_actions():
    fp = two_state_fixed_point(0.0, 0.0)
    assert fp["b_star"] == 0.0
    assert fp["P_survive"] == 0.0


@pytest.mark.parametrize("A_on,A_off", [(0.1, 1.0), (10.0, 0.1), (10.0, 10.0)])
def test_two_state_fixed_point_bounded_zero_one(A_on, A_off):
    fp = two_state_fixed_point(A_on, A_off)
    assert 0.0 <= fp["b_star"] <= 1.0
    assert 0.0 <= fp["P_survive"] <= 1.0


@pytest.mark.parametrize("preset_name,targets", REFERENCE_ACTION_PRESETS.items())
def test_reference_action_presets_solve_and_reproduce_target_action(preset_name, targets):
    A_on_ref, A_off_ref = targets
    cfg_template = CrackRebondingControls(
        bond_activation_volume_m3=1.5e-29,
        rupture_activation_volume_m3=1.5e-29,
        healing_cooperative_order=1.0,
    )
    resolved = solve_reference_action_barriers(
        A_on_ref,
        A_off_ref,
        reference_patch_distance_m=1.0e-7,
        reference_contact_radius_m=1.0e-8,
        cfg_template=cfg_template,
    )
    resolved.validate()
    assert math.isfinite(resolved.bond_barrier_eV)
    assert math.isfinite(resolved.rupture_barrier_eV)


def test_freeze_reference_action_preset_hash_is_deterministic():
    cfg_template = CrackRebondingControls(
        bond_activation_volume_m3=1.5e-29, rupture_activation_volume_m3=1.5e-29
    )
    kwargs = dict(
        cfg_template=cfg_template,
        reference_patch_distance_m=1.0e-7,
        reference_contact_radius_m=1.0e-8,
    )
    a = freeze_reference_action_preset("reversible", 1.0, 1.0, **kwargs)
    b = freeze_reference_action_preset("reversible", 1.0, 1.0, **kwargs)
    c = freeze_reference_action_preset("persistent", 10.0, 0.1, **kwargs)
    assert a["sha256"] == b["sha256"]
    assert a["sha256"] != c["sha256"]
