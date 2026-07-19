import csv
import json
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.frozen_geometry_load_invariance_v10213 import (
    _validate_coefficients,
)
from arrhenius_fracture.physical_fem_capture_v10213 import (
    PhysicalFEMCapture,
    load_extension_capture_requests,
)
from arrhenius_fracture.sharp_front_v10_2_13_capture import (
    _repair_capture_audits,
    _repair_multitemperature_geometry_summary,
)
from arrhenius_fracture.signed_kernel_family_v10213 import (
    SCHEMA,
    ExtensionOnlySigned2DShieldingKernelFamily,
)


def _artifact():
    states = []
    for index, extension in enumerate((0.0, 1.0e-5)):
        value = 1.0e6 * (1.0 + extension / 1.0e-5)
        states.append(
            {
                "state_id": f"E{index}",
                "r_eff_over_r0": 1.0,
                "opening_strength_fraction": 0.0,
                "crack_extension_m": extension,
                "active_kernel_I_Pa_sqrt_m_per_signed_line": [[value]],
                "wake_kernel_I_Pa_sqrt_m_per_signed_line": [[]],
                "active_kernel_II_Pa_sqrt_m_per_signed_line": [[0.0]],
                "wake_kernel_II_Pa_sqrt_m_per_signed_line": [[]],
            }
        )
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
            "neighbors": 2,
            "power": 2.0,
            "envelope_relative_tolerance": 1.0e-10,
        },
        "opening_boundary_policy": {"policy": "strict"},
        "kernel_radius_axis_policy": "disabled_constant_compatibility",
        "kernel_radius_compatibility_coordinate": 1.0,
        "opening_axis_policy": "validation_only_collapsed_constant_compatibility",
        "kernel_opening_compatibility_coordinate": 0.0,
        "finite_radius_fem_kernel_claimed": False,
        "same_kernel_family_for_monotonic_and_fatigue": True,
        "constitutive_K_shield_cap_present": False,
        "signed_burgers_population_required": True,
        "full_mpz_grid_values_are_spatial_projection": True,
        "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
        "frozen_geometry_load_invariance_passed": True,
        "production_parameterization_allowed": False,
        "states": states,
    }


def test_extension_only_family_ignores_runtime_opening_and_radius(tmp_path):
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps(_artifact()))
    family = ExtensionOnlySigned2DShieldingKernelFamily.from_json(path)
    first, _ = family.resolve(
        r_eff_over_r0=1.0,
        opening_strength_fraction=0.05,
        crack_extension_m=0.0,
    )
    second, _ = family.resolve(
        r_eff_over_r0=50.0,
        opening_strength_fraction=0.95,
        crack_extension_m=0.0,
    )
    assert np.allclose(first, second)
    audit = family.audit_payload()
    assert audit["opening_strength_fraction_used_for_interpolation"] is False
    assert audit["last_observed_opening_strength_fraction"] == pytest.approx(0.95)
    assert audit["last_query"][1] == pytest.approx(0.0)


def test_extension_only_family_remains_fail_closed_in_path_extension(tmp_path):
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps(_artifact()))
    family = ExtensionOnlySigned2DShieldingKernelFamily.from_json(path)
    with pytest.raises(RuntimeError, match="crack_extension_m"):
        family.resolve(
            r_eff_over_r0=2.0,
            opening_strength_fraction=0.5,
            crack_extension_m=2.0e-5,
        )


def _response_rows(load_coefficient):
    rows = []
    for scale, coefficient in load_coefficient.items():
        for sign in (-1, 1):
            for magnitude in (0.25, 0.5):
                delta = sign * magnitude
                rows.append(
                    {
                        "region": "active",
                        "system": 0,
                        "bin": 0,
                        "x_m": 0.5,
                        "load_scale": scale,
                        "delta_signed_line_content": delta,
                        "K_I_base_Pa_sqrt_m": 1.0e7 * scale,
                        "K_I_perturbed_Pa_sqrt_m": 1.0e7 * scale - coefficient * delta,
                        "K_II_base_Pa_sqrt_m": 0.0,
                        "K_II_perturbed_Pa_sqrt_m": 0.0,
                    }
                )
    return rows


def test_frozen_geometry_load_invariance_check_passes_and_fails():
    passed = _validate_coefficients(
        _response_rows({0.5: 2.0e5, 1.0: 2.0e5, 1.5: 2.0e5}),
        linearity_tolerance=0.03,
        load_invariance_tolerance=0.05,
        significance_floor_fraction=1.0e-3,
    )
    assert passed["within_load_sign_amplitude_linearity_passed"]
    assert passed["frozen_geometry_load_invariance_passed"]
    failed = _validate_coefficients(
        _response_rows({0.5: 2.0e5, 1.0: 2.0e5, 1.5: 2.5e5}),
        linearity_tolerance=0.03,
        load_invariance_tolerance=0.05,
        significance_floor_fraction=1.0e-3,
    )
    assert not failed["frozen_geometry_load_invariance_passed"]


def test_extension_request_table_does_not_require_opening_or_radius(tmp_path):
    path = tmp_path / "states.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "state_id",
                "temperature_K",
                "cumulative_crack_path_extension_m",
                "extension_tolerance_m",
                "interaction_ell_m",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_id": "E200",
                "temperature_K": 700,
                "cumulative_crack_path_extension_m": 200e-6,
                "extension_tolerance_m": 2.5e-6,
                "interaction_ell_m": 2.0e-6,
            }
        )
    requests = load_extension_capture_requests(path)
    assert len(requests) == 1
    assert requests[0].crack_extension_m == pytest.approx(200e-6)
    capture = PhysicalFEMCapture(requests, tmp_path / "capture")
    match = capture._matching_request(
        700.0,
        {
            "r_eff_over_r0": 500.0,
            "opening_strength_fraction": 0.99,
            "crack_extension_m": 201e-6,
        },
    )
    assert match is requests[0]


def test_multitemperature_geometry_summary_is_assigned_only_to_last_temperature(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            [
                {"T": 300.0, "n_geometry_events": 999},
                {"T": 1200.0},
            ]
        )
    )
    (root / "run_args.json").write_text(json.dumps({"temperatures": [300.0, 1200.0]}))
    geometry = [
        {
            "event_advance_m": 5.0e-6,
            "requested_fixed_length_m": 5.0e-6,
            "x0": 0.5e-3,
            "x1": 0.5035e-3,
        },
        {
            "event_advance_m": 5.0e-6,
            "requested_fixed_length_m": 5.0e-6,
            "x0": 0.5035e-3,
            "x1": 0.5070e-3,
        },
    ]
    (root / "stochastic_avalanche_geometry_events.json").write_text(json.dumps(geometry))
    (root / "sharp_wake_advance_log.csv").write_text("event_index\n1\n2\n")
    repair = _repair_multitemperature_geometry_summary(root)
    summary = json.loads((root / "summary.json").read_text())
    assert repair["geometry_diagnostics_temperature_K"] == 1200.0
    assert "n_geometry_events" not in summary[0]
    assert summary[1]["n_geometry_events"] == 2
    assert summary[1]["geometry_diagnostics_temperature_K"] == 1200.0
    assert (root / "stochastic_avalanche_geometry_events_1200K.json").is_file()


def test_capture_audit_rewrite_removes_stale_cap_claims(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    for name in (
        "v10_1_driver_modes.json",
        "v10_1_1_source_model.json",
        "v10_1_7_5_transport_mode.json",
    ):
        (root / name).write_text(
            json.dumps(
                {
                    "manifest_K_shield_cap_enabled": True,
                    "campaign_active_shielding_cap_preserved": True,
                }
            )
        )
    _repair_capture_audits(root)
    for name in (
        "v10_1_driver_modes.json",
        "v10_1_1_source_model.json",
        "v10_1_7_5_transport_mode.json",
    ):
        payload = json.loads((root / name).read_text())
        assert payload["manifest_K_shield_cap_enabled"] is False
        assert payload["campaign_active_shielding_cap_preserved"] is False
        assert payload["constitutive_K_shield_cap_applied"] is False
