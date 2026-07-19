import json
from pathlib import Path

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
        "fixed_kernel_assessment": {},
        "interpolation": {},
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
