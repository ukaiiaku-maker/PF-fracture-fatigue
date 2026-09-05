import json
import copy
from pathlib import Path
import pickle

import numpy as np
import pytest

from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    SCHEMA,
)


def _payload():
    active_x = [0.5e-6, 1.5e-6]
    wake_x = [0.5e-6]
    states = []
    for state_id, extension, scale in (("E000", 0.0, 1.0), ("E200", 2e-4, 1.2)):
        states.append(
            {
                "state_id": state_id,
                "r_eff_over_r0": 1.0,
                "opening_strength_fraction": 0.0,
                "crack_extension_m": extension,
                "active_kernel_I_Pa_sqrt_m_per_signed_line": [
                    [scale, 0.5 * scale],
                    [-0.25 * scale, 0.1 * scale],
                ],
                "wake_kernel_I_Pa_sqrt_m_per_signed_line": [[0.0], [0.0]],
                "active_kernel_II_Pa_sqrt_m_per_signed_line": [
                    [0.1 * scale, 0.05 * scale],
                    [0.2 * scale, 0.1 * scale],
                ],
                "wake_kernel_II_Pa_sqrt_m_per_signed_line": [[0.0], [0.0]],
            }
        )
    return {
        "schema": SCHEMA,
        "states": states,
        "state_axes": [
            "r_eff_over_r0",
            "opening_strength_fraction",
            "crack_extension_m",
        ],
        "active_x_m": active_x,
        "wake_x_m": wake_x,
        "activation_to_line_content_by_system": [1.0, 1.0],
        "source_capacity_bounds_per_system": [[0.0, 100.0], [0.0, 100.0]],
        "fixed_kernel_assessment": {"fixed_kernel_accepted": False},
        "interpolation": {"method": "inverse_distance"},
        "opening_boundary_policy": {"policy": "strict"},
        "candidate_independent": True,
        "counts_are_signed_burgers_lines": True,
        "kernel_from_2d_unit_signed_perturbations": True,
        "kernel_from_signed_interaction_integral": True,
        "analytic_auxiliary_gradients": True,
        "hermite_domain_weight": True,
        "positive_and_negative_perturbations": True,
        "multiple_perturbation_magnitudes": True,
        "multi_amplitude_validation_passed": True,
        "normalization_is_mechanically_derived": True,
        "fitted_attenuation_factor": False,
        "constitutive_K_shield_cap": False,
        "complete_cartesian_state_grid": True,
        "kernel_source": "2d_unit_signed_slip_perturbation",
        "normalization_source": "process_zone_geometry_and_line_spacing",
        "kernel_radius_axis_policy": "disabled_constant_compatibility",
        "opening_axis_policy": "validation_only_collapsed_constant_compatibility",
        "same_kernel_family_for_monotonic_and_fatigue": True,
        "constitutive_K_shield_cap_present": False,
        "signed_burgers_population_required": True,
        "full_mpz_grid_values_are_spatial_projection": True,
        "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
        "frozen_geometry_load_invariance_passed": True,
        "active_kernel_mechanically_measured": True,
        "wake_kernel_mechanically_measured": False,
        "wake_shielding_supported": False,
        "kernel_opening_compatibility_coordinate": 0.0,
    }


def test_active_only_loader_accepts_zero_wake(tmp_path: Path):
    path = tmp_path / "active.json"
    path.write_text(json.dumps(_payload()))
    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(path)
    assert all(np.allclose(state.wake_I, 0.0) for state in family.states)
    assert all(np.allclose(state.wake_II, 0.0) for state in family.states)
    audit = family.audit_payload()
    assert audit["wake_shielding_supported"] is False
    assert audit["wake_kernel_forced_zero"] is True


def test_active_only_loader_rejects_nonzero_wake(tmp_path: Path):
    payload = _payload()
    payload["states"][0]["wake_kernel_I_Pa_sqrt_m_per_signed_line"][0][0] = 1.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="zero wake kernels"):
        ActiveOnlySigned2DShieldingKernelFamily.from_json(path)


def test_append_only_policy_preserves_exact_legacy_domain(tmp_path: Path):
    old_payload = _payload()
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old_payload))
    old = ActiveOnlySigned2DShieldingKernelFamily.from_json(old_path)

    new_payload = json.loads(json.dumps(old_payload))
    for state_id, extension, scale in (
        ("E210", 2.1e-4, 2.0),
        ("E400", 4.0e-4, 3.0),
    ):
        row = json.loads(json.dumps(new_payload["states"][-1]))
        row["state_id"] = state_id
        row["crack_extension_m"] = extension
        for key in (
            "active_kernel_I_Pa_sqrt_m_per_signed_line",
            "active_kernel_II_Pa_sqrt_m_per_signed_line",
        ):
            row[key] = (scale * np.asarray(row[key], dtype=float)).tolist()
        new_payload["states"].append(row)
    new_payload["append_only_legacy_domain_policy"] = {
        "model_id": "v10.2.14_exact_legacy_domain_prefix_v1",
        "legacy_domain_max_crack_extension_m": 2.0e-4,
        "legacy_state_ids": ["E000", "E200"],
    }
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps(new_payload))
    new = ActiveOnlySigned2DShieldingKernelFamily.from_json(new_path)

    for extension in np.linspace(0.0, 2.0e-4, 41):
        old_active, old_wake = old.resolve(
            r_eff_over_r0=3.0,
            opening_strength_fraction=0.75,
            crack_extension_m=float(extension),
        )
        new_active, new_wake = new.resolve(
            r_eff_over_r0=3.0,
            opening_strength_fraction=0.75,
            crack_extension_m=float(extension),
        )
        assert np.array_equal(new_active, old_active)
        assert np.array_equal(new_wake, old_wake)
        assert np.array_equal(new.active_kernel_II, old.active_kernel_II)
        assert np.array_equal(new.wake_kernel_II, old.wake_kernel_II)
    assert new.audit_payload()["append_only_legacy_domain_active"] is True
    new.resolve(
        r_eff_over_r0=1.0,
        opening_strength_fraction=0.0,
        crack_extension_m=3.0e-4,
    )

    for variant in (
        new.clone_for_engine(), copy.deepcopy(new),
        pickle.loads(pickle.dumps(new, protocol=5)),
    ):
        old_active, _ = old.resolve(
            r_eff_over_r0=1.75,
            opening_strength_fraction=0.4,
            crack_extension_m=1.0e-4,
        )
        new_active, _ = variant.resolve(
            r_eff_over_r0=1.75,
            opening_strength_fraction=0.4,
            crack_extension_m=1.0e-4,
        )
        assert np.array_equal(new_active, old_active)
        assert np.array_equal(variant._last_weights[:2], old._last_weights)
        assert np.array_equal(variant._last_weights[2:], np.zeros(2))


def test_nested_append_only_policy_preserves_both_qualified_prefixes(tmp_path: Path):
    base = _payload()
    first = json.loads(json.dumps(base))
    for state_id, extension, scale in (
        ("E300", 3.0e-4, 2.0), ("E400", 4.0e-4, 3.0),
    ):
        row = json.loads(json.dumps(first["states"][-1]))
        row["state_id"] = state_id
        row["crack_extension_m"] = extension
        for key in (
            "active_kernel_I_Pa_sqrt_m_per_signed_line",
            "active_kernel_II_Pa_sqrt_m_per_signed_line",
        ):
            row[key] = (scale * np.asarray(row[key], dtype=float)).tolist()
        first["states"].append(row)
    first_policy = {
        "model_id": "v10.2.14_exact_legacy_domain_prefix_v1",
        "legacy_domain_max_crack_extension_m": 2.0e-4,
        "legacy_state_ids": ["E000", "E200"],
    }
    first["append_only_legacy_domain_policy"] = first_policy
    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps(first))
    first_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(first_path)

    second = json.loads(json.dumps(first))
    row = json.loads(json.dumps(second["states"][-1]))
    row["state_id"] = "E600"
    row["crack_extension_m"] = 6.0e-4
    second["states"].append(row)
    second["append_only_legacy_domain_policy"] = {
        "model_id": "v10.2.14_exact_legacy_domain_prefix_v1",
        "legacy_domain_max_crack_extension_m": 4.0e-4,
        "legacy_state_ids": ["E000", "E200", "E300", "E400"],
        "legacy_append_only_legacy_domain_policy": first_policy,
    }
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps(second))
    second_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(second_path)

    for extension in np.linspace(0.0, 4.0e-4, 81):
        kwargs = dict(
            r_eff_over_r0=2.0,
            opening_strength_fraction=0.5,
            crack_extension_m=float(extension),
        )
        first_active, first_wake = first_family.resolve(**kwargs)
        second_active, second_wake = second_family.resolve(**kwargs)
        assert np.array_equal(second_active, first_active)
        assert np.array_equal(second_wake, first_wake)
        assert np.array_equal(
            second_family.active_kernel_II, first_family.active_kernel_II
        )


def test_append_only_boundary_tolerance_and_upper_bound_fail_closed(tmp_path: Path):
    payload = _payload()
    for state_id, extension in (("E300", 3.0e-4), ("E400", 4.0e-4)):
        row = json.loads(json.dumps(payload["states"][-1]))
        row["state_id"] = state_id
        row["crack_extension_m"] = extension
        payload["states"].append(row)
    payload["interpolation"]["envelope_relative_tolerance"] = 1.0e-10
    payload["append_only_legacy_domain_policy"] = {
        "model_id": "v10.2.14_exact_legacy_domain_prefix_v1",
        "legacy_domain_max_crack_extension_m": 2.0e-4,
        "legacy_state_ids": ["E000", "E200"],
    }
    path = tmp_path / "bounded.json"
    path.write_text(json.dumps(payload))
    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(path)
    family.resolve(
        r_eff_over_r0=1.0, opening_strength_fraction=0.0,
        crack_extension_m=2.0e-4 + 1.0e-10,
    )
    assert np.array_equal(family._last_weights[2:], np.zeros(2))
    with pytest.raises(RuntimeError, match="outside"):
        family.resolve(
            r_eff_over_r0=1.0, opening_strength_fraction=0.0,
            crack_extension_m=4.0e-4 + 2.0e-10,
        )


def test_append_only_policy_rejects_missing_or_overlapping_states(tmp_path: Path):
    payload = _payload()
    payload["append_only_legacy_domain_policy"] = {
        "model_id": "v10.2.14_exact_legacy_domain_prefix_v1",
        "legacy_domain_max_crack_extension_m": 2.0e-4,
        "legacy_state_ids": ["E000", "missing"],
    }
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing legacy state"):
        ActiveOnlySigned2DShieldingKernelFamily.from_json(path)

    payload = _payload()
    payload["append_only_legacy_domain_policy"] = {
        "model_id": "v10.2.14_exact_legacy_domain_prefix_v1",
        "legacy_domain_max_crack_extension_m": 2.0e-4,
        "legacy_state_ids": ["E000", "E200"],
    }
    path = tmp_path / "no-new-state.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="strictly beyond"):
        ActiveOnlySigned2DShieldingKernelFamily.from_json(path)
