from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.anisotropic_emission_v10174 import (
    AnisotropicEmissionConfig,
    _anisotropic_campaign_emit,
    finite_source_emission_update,
    probe_tensor_ahead,
    resolve_channel_drives,
)


def _rotate_tensor(tensor, angle_deg):
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return rotation @ tensor @ rotation.T


def test_hydrostatic_tensor_has_zero_slip_drive():
    tensor = 2.0e9 * np.eye(2)
    drive = resolve_channel_drives(
        tensor,
        [tensor, tensor],
        crystal_theta_deg=45.0,
        schmid_reference=0.5,
    )
    assert np.allclose(drive["tau_signed_Pa"], 0.0, atol=1.0e-6)
    assert np.allclose(drive["drive_factors"], 0.0, atol=1.0e-15)
    assert drive["factors_normalized"] is False
    assert drive["factors_clipped"] is False


def test_joint_crystal_and_tensor_rotation_preserves_channel_drives():
    tensor = np.array([[2.0e9, 0.6e9], [0.6e9, 2.0e9]])
    initial = resolve_channel_drives(
        tensor,
        [tensor, tensor],
        crystal_theta_deg=15.0,
        schmid_reference=0.5,
    )
    rotated = _rotate_tensor(tensor, 90.0)
    transformed = resolve_channel_drives(
        rotated,
        [rotated, rotated],
        crystal_theta_deg=105.0,
        schmid_reference=0.5,
    )
    assert np.allclose(
        np.abs(initial["tau_signed_Pa"]),
        np.abs(transformed["tau_signed_Pa"]),
        rtol=1.0e-12,
        atol=1.0e-6,
    )
    assert np.allclose(
        initial["drive_factors"],
        transformed["drive_factors"],
        rtol=1.0e-12,
        atol=1.0e-15,
    )


def test_finite_source_update_is_exact_and_bounded():
    available = np.array([7.0, 11.0])
    rates = np.array([0.3, 1.7])
    emitted, probability = finite_source_emission_update(
        available, rates, dt=0.4
    )
    expected_probability = 1.0 - np.exp(-rates * 0.4)
    assert np.allclose(probability, expected_probability)
    assert np.allclose(emitted, available * expected_probability)
    assert np.all(emitted >= 0.0)
    assert np.all(emitted <= available)


def _fake_campaign_state():
    n_systems = 2
    n_bins = 4
    cfg = SimpleNamespace(
        blunting_length_m=1.0,
        taylor_stress_fraction=0.5,
        source_bin_count=1,
    )
    state = SimpleNamespace(
        n_systems=n_systems,
        n_bins=n_bins,
        cfg=cfg,
        x=np.arange(n_bins, dtype=float) + 0.5,
        dx=1.0,
        mobile=np.zeros((n_systems, n_bins)),
        retained=np.zeros((n_systems, n_bins)),
        accumulated_slip=np.zeros((n_systems, n_bins)),
        available_sites=np.array([10.0, 10.0]),
        site_capacity=np.array([10.0, 10.0]),
        tip_source_activity=np.ones(n_systems),
        emitted_total=0.0,
        _campaign_b=1.0,
        _campaign_G_Pa=0.0,
        _campaign_backstress_scale=1.0,
        _anisotropic_drive_factors=np.array([0.5, 1.5]),
        emission_rate_per_site=lambda sigma, _T: 1.0e-3 * sigma,
    )
    return state


def test_anisotropic_factor_enters_before_barrier_without_second_weight():
    state = _fake_campaign_state()
    emitted = _anisotropic_campaign_emit(
        state,
        dt=0.01,
        stress_Pa=100.0,
        T_K=700.0,
    )
    rates = np.array([0.05, 0.15])
    expected_probability = 1.0 - np.exp(-rates * 0.01)
    expected = 10.0 * expected_probability
    assert np.allclose(state.anisotropic_last_lambda_emit_by_system_s, rates)
    assert np.allclose(state.anisotropic_last_dN_emit_by_system, expected)
    assert np.isclose(emitted, np.sum(expected))
    # A post-hazard f/max(f) multiplier would reduce the first channel by 1/3.
    assert not np.isclose(
        state.anisotropic_last_dN_emit_by_system[0],
        expected[0] / 3.0,
    )


def test_nontrivial_post_hazard_weights_are_rejected():
    state = _fake_campaign_state()
    with pytest.raises(RuntimeError, match="post-hazard"):
        _anisotropic_campaign_emit(
            state,
            dt=0.01,
            stress_Pa=100.0,
            T_K=700.0,
            system_weights=np.array([0.5, 1.0]),
        )


def _triangle_at(center):
    x, y = center
    eps = 0.1e-6
    return np.array(
        [[x - eps, y - eps], [x + eps, y - eps], [x, y + 2.0 * eps]]
    )


def test_tensor_probe_excludes_damaged_high_stress_element():
    centers = [
        (8.0e-6, 0.0),
        (10.0e-6, 1.0e-6),
        (12.0e-6, -1.0e-6),
        (10.0e-6, 0.0),
    ]
    nodes = np.vstack([_triangle_at(center) for center in centers])
    elems = np.arange(12, dtype=int).reshape(4, 3)
    mesh = SimpleNamespace(
        nodes=nodes,
        elems=elems,
        area_e=np.ones(4),
        ne=4,
        nn=12,
    )
    sigma = np.array(
        [
            [1.0e9, 1.0e9, 1.0e9, 100.0e9],
            [2.0e9, 2.0e9, 2.0e9, 100.0e9],
            [0.2e9, 0.2e9, 0.2e9, 100.0e9],
        ]
    )
    damage = np.array([0.0, 0.0, 0.0, 1.0])
    config = AnisotropicEmissionConfig(
        probe_radius_m=10.0e-6,
        sector_half_angle_deg=25.0,
        damage_cutoff=0.85,
        min_elements=3,
    )
    probe = probe_tensor_ahead(
        mesh,
        sigma,
        damage,
        tip_xy=np.array([0.0, 0.0]),
        ray_direction=np.array([1.0, 0.0]),
        config=config,
    )
    assert probe["reliable"] is True
    tensor = np.asarray(probe["tensor"])
    assert np.isclose(tensor[0, 0], 1.0e9)
    assert np.isclose(tensor[1, 1], 2.0e9)
    assert np.isclose(tensor[0, 1], 0.2e9)
