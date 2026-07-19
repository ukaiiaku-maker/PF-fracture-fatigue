import json

import numpy as np
import pytest

from arrhenius_fracture.signed_kernel_family_v10212 import (
    SCHEMA,
    RealSigned2DShieldingKernelFamily,
)


def _artifact():
    states = []
    index = 0
    for opening in (0.0, 0.5, 1.0):
        for extension in (0.0, 1.0e-5):
            value = 1.0e6 * (1.0 + opening + extension / 1.0e-5)
            states.append(
                {
                    "state_id": f"S{index:02d}",
                    "r_eff_over_r0": 1.0,
                    "opening_strength_fraction": opening,
                    "crack_extension_m": extension,
                    "active_kernel_I_Pa_sqrt_m_per_signed_line": [[value]],
                    "wake_kernel_I_Pa_sqrt_m_per_signed_line": [[]],
                    "active_kernel_II_Pa_sqrt_m_per_signed_line": [[0.0]],
                    "wake_kernel_II_Pa_sqrt_m_per_signed_line": [[]],
                }
            )
            index += 1
    return {
        "schema": SCHEMA,
        "candidate_independent": True,
        "counts_are_signed_burgers_lines": True,
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
        "state_axes": [
            "r_eff_over_r0",
            "opening_strength_fraction",
            "crack_extension_m",
        ],
        "normalization_source": "process_zone_geometry_and_line_spacing",
        "active_x_m": [0.5],
        "wake_x_m": [],
        "activation_to_line_content_by_system": [1.0],
        "source_capacity_bounds_per_system": [[1.0, 10.0]],
        "fixed_kernel_assessment": {"fixed_kernel_accepted": False},
        "interpolation": {
            "method": "inverse_distance",
            "neighbors": 4,
            "power": 2.0,
            "envelope_relative_tolerance": 1.0e-10,
        },
        "opening_boundary_policy": {
            "policy": "strict",
            "lower_boundary_validated": False,
            "upper_boundary_validated": False,
        },
        "kernel_radius_axis_policy": "disabled_constant_compatibility",
        "kernel_radius_compatibility_coordinate": 1.0,
        "finite_radius_fem_kernel_claimed": False,
        "same_kernel_family_for_monotonic_and_fatigue": True,
        "constitutive_K_shield_cap_present": False,
        "signed_burgers_population_required": True,
        "full_mpz_grid_values_are_spatial_projection": True,
        "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
        "production_parameterization_allowed": False,
        "states": states,
    }


def test_loader_accepts_one_radius_opening_extension_grid(tmp_path):
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps(_artifact()))
    family = RealSigned2DShieldingKernelFamily.from_json(path)
    first, _ = family.resolve(
        r_eff_over_r0=1.0,
        opening_strength_fraction=0.5,
        crack_extension_m=0.0,
    )
    second, _ = family.resolve(
        r_eff_over_r0=250.0,
        opening_strength_fraction=0.5,
        crack_extension_m=0.0,
    )
    assert np.allclose(first, second)
    audit = family.audit_payload()
    assert audit["kernel_radius_axis_policy"] == "disabled_constant_compatibility"
    assert audit["last_observed_analytical_r_eff_over_r0"] == pytest.approx(250.0)
    assert audit["last_query"][0] == pytest.approx(1.0)
    assert audit["analytical_r_eff_used_for_interpolation"] is False


def test_extension_remains_fail_closed_even_when_radius_axis_is_disabled(tmp_path):
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps(_artifact()))
    family = RealSigned2DShieldingKernelFamily.from_json(path)
    with pytest.raises(RuntimeError, match="crack_extension_m"):
        family.resolve(
            r_eff_over_r0=10.0,
            opening_strength_fraction=0.5,
            crack_extension_m=2.0e-5,
        )


def test_loader_rejects_a_second_radius_level(tmp_path):
    payload = _artifact()
    payload["states"][0]["r_eff_over_r0"] = 1.2
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="constant radius"):
        RealSigned2DShieldingKernelFamily.from_json(path)
