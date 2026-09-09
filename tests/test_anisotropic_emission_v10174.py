from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.anisotropic_emission_v10174 import (
    AnisotropicEmissionConfig,
    _anisotropic_campaign_emit,
    _tp_state_diagnostics_enabled,
    _tp_state_diagnostics_mode,
    finite_source_emission_update,
    probe_tensor_ahead,
    require_admissible_tensor_drive,
    resolve_channel_drives,
    tensor_normalization_admissibility,
    bind_explicit_accepted_tensor_drive,
    evaluate_tensor_conditioning_sentinel_cases,
    AnisotropicStochasticAvalancheTipEngine,
    OBSERVER,
)


def test_taylor_peierls_state_observer_is_default_off(monkeypatch):
    monkeypatch.delenv("ONED_V2_TP_STATE_DIAGNOSTICS", raising=False)
    assert _tp_state_diagnostics_enabled() is False
    monkeypatch.setenv("ONED_V2_TP_STATE_DIAGNOSTICS", "1")
    assert _tp_state_diagnostics_enabled() is True
    monkeypatch.setenv("ONED_V2_TP_STATE_DIAGNOSTICS", "false")
    assert _tp_state_diagnostics_enabled() is False
    monkeypatch.setenv("ONED_V2_TP_STATE_DIAGNOSTICS", "events")
    assert _tp_state_diagnostics_enabled() is True
    assert _tp_state_diagnostics_mode() == "events"


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


@pytest.mark.parametrize(
    "name,tensor,reliable,admissible,reason",
    [
        ("tensile", [[1.0e6, 2.0e5], [2.0e5, 3.0e6]], True, True, None),
        ("compressive", [[-2.0e6, 1.0e5], [1.0e5, -1.0e6]], True, False,
         "nonpositive_tensile_opening_scale_for_anisotropic_normalization"),
        ("near_zero_positive", [[1.0e-12, 0.0], [0.0, 2.0e-12]], True, True, None),
        ("unreliable", [[1.0e6, 0.0], [0.0, 2.0e6]], False, False,
         "unreliable_tensor_probe_for_anisotropic_normalization"),
        ("nonfinite", [[float("nan"), 0.0], [0.0, 1.0]], True, False,
         "nonfinite_tensor_for_anisotropic_normalization"),
    ],
)
def test_v4_tensor_normalization_is_fail_closed_without_one_pa_floor(
    name, tensor, reliable, admissible, reason,
):
    result = tensor_normalization_admissibility(
        tensor, [tensor, tensor], probe_reliable=reliable,
    )
    assert result["tensor_drive_admissible"] is admissible, name
    assert result["tensor_drive_rejection_reason"] == reason
    assert result["opening_normalization_floor_active"] is False
    if admissible:
        assert result["drive_factors"] is not None
    else:
        assert result["drive_factors"] is None
        with pytest.raises(RuntimeError, match=reason):
            require_admissible_tensor_drive(result)


def test_v4_mixed_tensor_uses_positive_principal_opening_without_cap_or_clipping():
    tensor = np.array([[-2.0e6, 3.0e6], [3.0e6, -1.0e6]])
    result = tensor_normalization_admissibility(tensor, [tensor, -tensor])
    assert result["sigma_nn_probe_Pa"] < 0.0
    assert result["sigma1_probe_Pa"] > 0.0
    assert result["tensor_drive_admissible"] is True
    assert result["drive_factors"][0] == result["drive_factors"][1]


def test_v4_tensor_admissibility_is_order_invariant_and_state_pure():
    opening = np.array([[2.0e6, 3.0e5], [3.0e5, 4.0e6]])
    channels = [np.array([[0.0, 2.0e5], [2.0e5, 0.0]]),
                np.array([[0.0, -7.0e5], [-7.0e5, 0.0]])]
    before = [value.copy() for value in channels]
    forward = tensor_normalization_admissibility(opening, channels)
    reverse = tensor_normalization_admissibility(opening, tuple(reversed(channels)))
    assert forward["drive_factors"] == list(reversed(reverse["drive_factors"]))
    assert all(np.array_equal(a, b) for a, b in zip(channels, before))


def test_positive_j_does_not_override_invalid_tensor_drive():
    drive = tensor_normalization_admissibility(
        [[-2.0, 0.0], [0.0, -1.0]],
        [[[-2.0, 0.0], [0.0, -1.0]]] * 2,
    )
    drive["positive_kinetic_J_J_per_m2"] = 1.0e12
    with pytest.raises(
        RuntimeError,
        match="nonpositive_tensile_opening_scale_for_anisotropic_normalization",
    ):
        require_admissible_tensor_drive(drive)


def test_contradictory_reliable_and_admissible_flags_fail_closed():
    drive = {
        "reliable": True,
        "tensor_drive_reliable": False,
        "tensor_drive_admissible": False,
        "drive_factors": [None, None],
        "tensor_drive_rejection_reason": "opening_scale_not_resolved_above_probe_uncertainty",
    }
    with pytest.raises(RuntimeError, match="opening_scale_not_resolved"):
        require_admissible_tensor_drive(drive)


def test_legacy_observer_adoption_rejects_before_installing_invalid_factors():
    prior = {"accepted": "unchanged"}
    calls = []
    fake = SimpleNamespace(
        _anisotropic_drive_serial=-1,
        _anisotropic_drive=prior,
        _install_current_drive_on_state=lambda: calls.append("installed"),
    )
    OBSERVER.latest_drive = {
        "drive_serial": 4,
        "reliable": False,
        "tensor_drive_reliable": False,
        "tensor_drive_admissible": False,
        "drive_factors": [None, None],
        "tensor_drive_rejection_reason": "opening_scale_not_resolved_above_probe_uncertainty",
    }
    with pytest.raises(RuntimeError, match="opening_scale_not_resolved"):
        AnisotropicStochasticAvalancheTipEngine._adopt_latest_drive(fake)
    assert fake._anisotropic_drive is prior
    assert fake._anisotropic_drive_serial == -1
    assert calls == []


def test_functional_conditioning_sentinel_executes_production_gate():
    cases = [{
        "case": "tensile",
        "opening_tensor_Pa": [[2.0e6, 0.0], [0.0, 3.0e6]],
        "channel_tensors_Pa": [
            [[0.0, 2.0e5], [2.0e5, 0.0]],
            [[0.0, -3.0e5], [-3.0e5, 0.0]],
        ],
        "expected_accepted": True,
    }, {
        "case": "compressive",
        "opening_tensor_Pa": [[-2.0e6, 0.0], [0.0, -3.0e6]],
        "channel_tensors_Pa": [
            [[0.0, 2.0e5], [2.0e5, 0.0]],
            [[0.0, -3.0e5], [-3.0e5, 0.0]],
        ],
        "expected_accepted": False,
        "expected_rejection_reason": "nonpositive_tensile_opening_scale_for_anisotropic_normalization",
    }]
    records = evaluate_tensor_conditioning_sentinel_cases(cases)
    assert all(row["passed"] for row in records)
    assert records[0]["conditioning_qualification_scope"] == "opening_denominator_only"


def test_near_resolution_opening_with_finite_channel_shear_fails_closed():
    opening = np.array([[1.0e-12, 0.0], [0.0, 2.0e-12]])
    channels = [
        np.array([[0.0, 1.0e6], [1.0e6, 0.0]]),
        np.array([[0.0, -2.0e6], [-2.0e6, 0.0]]),
    ]
    state = {"accepted": "unchanged"}
    rng = np.random.default_rng(3621)
    state_before = repr(state)
    rng_before = repr(rng.bit_generator.state)

    result = tensor_normalization_admissibility(opening, channels)

    assert result["sigma_amplitude_Pa"] == pytest.approx(2.0e-12)
    assert result["probe_uncertainty_Pa"] > result["sigma_amplitude_Pa"]
    assert result["opening_to_uncertainty_ratio"] < 1.0
    assert result["tensor_drive_admissible"] is False
    assert result["drive_factors"] is None
    assert result["tensor_drive_rejection_reason"] == (
        "opening_scale_not_resolved_above_probe_uncertainty"
    )
    with pytest.raises(RuntimeError, match="opening_scale_not_resolved"):
        require_admissible_tensor_drive(result)
    assert repr(state) == state_before
    assert repr(rng.bit_generator.state) == rng_before


def test_explicit_accepted_tensor_binding_ignores_overwritten_global_observer(monkeypatch):
    stale = {"stress_field_state_id": "trial", "tensor_drive_admissible": True}
    OBSERVER.latest_drive = stale
    accepted = {
        "tensor_drive_admissible": True,
        "tensor_drive_reliable": True,
        "reliable": True,
        "drive_factors": [0.2, 0.4],
        "tensor_drive_rejection_reason": None,
    }
    monkeypatch.setattr(
        "arrhenius_fracture.anisotropic_emission_v10174.build_front_drive",
        lambda mesh, sigma_gp, damage, tip_xy, config: dict(accepted),
    )
    mesh = object()
    sigma = np.array([[1.0]])
    damage = np.array([0.0])
    rng = np.random.default_rng(3621)
    rng_before = repr(rng.bit_generator.state)
    result = bind_explicit_accepted_tensor_drive(
        mesh=mesh, sigma_gp=sigma, damage=damage, tip_xy=[1.0, 2.0],
        config=AnisotropicEmissionConfig(), accepted_state_id="accepted",
        stress_field_state_id="accepted-stress",
    )
    assert result["explicit_bound_accepted_state"] is True
    assert result["accepted_state_id"] == "accepted"
    assert result["stress_field_state_id"] == "accepted-stress"
    assert OBSERVER.latest_drive is result
    assert OBSERVER.latest_drive is not stale
    assert np.array_equal(sigma, [[1.0]])
    assert np.array_equal(damage, [0.0])
    assert repr(rng.bit_generator.state) == rng_before


def test_explicit_invalid_tensor_rejects_before_observer_or_rng_mutation(monkeypatch):
    prior = {"stress_field_state_id": "accepted-prior"}
    OBSERVER.latest_drive = prior
    monkeypatch.setattr(
        "arrhenius_fracture.anisotropic_emission_v10174.build_front_drive",
        lambda *args, **kwargs: {
            "tensor_drive_admissible": False,
            "tensor_drive_reliable": False,
            "reliable": False,
            "drive_factors": None,
            "tensor_drive_rejection_reason": (
                "nonpositive_tensile_opening_scale_for_anisotropic_normalization"
            ),
        },
    )
    rng = np.random.default_rng(3621)
    before = repr(rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="nonpositive_tensile"):
        bind_explicit_accepted_tensor_drive(
            mesh=object(), sigma_gp=np.array([[1.0]]), damage=np.array([0.0]),
            tip_xy=[0.0, 0.0], config=AnisotropicEmissionConfig(),
            accepted_state_id="accepted", stress_field_state_id="stress",
        )
    assert OBSERVER.latest_drive is prior
    assert repr(rng.bit_generator.state) == before


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
