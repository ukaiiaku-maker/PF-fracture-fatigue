import json
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.state_resolved_drive_family_v1027 import (
    StateResolvedSignedDriveFamily,
)
from arrhenius_fracture.state_resolved_reduced_campaign_v1027 import (
    DEFAULT_TEMPERATURES_K,
    score_ceramic_reference,
    score_dbtt,
    score_weakt,
)


def _write_drive_family(tmp_path):
    path = tmp_path / "drive.json"
    payload = {
        "schema": "v10.2.7_state_resolved_signed_drive_family",
        "candidate_independent": True,
        "derived_from_2d_tensor_probe": True,
        "signed_resolved_shear": True,
        "normalized_by_local_opening_stress": True,
        "fitted_to_toughness_or_fatigue": False,
        "production_parameterization_allowed": True,
        "state_axes": [
            "r_eff_over_r0",
            "opening_strength_fraction",
            "crack_extension_m",
        ],
        "interpolation": {
            "method": "inverse_distance",
            "neighbors": 2,
            "power": 2.0,
            "envelope_relative_tolerance": 1.0e-10,
        },
        "states": [
            {
                "state_id": "low",
                "r_eff_over_r0": 1.0,
                "opening_strength_fraction": 0.0,
                "crack_extension_m": 0.0,
                "signed_tau_over_sigma_by_system": [0.10, -0.05],
            },
            {
                "state_id": "high",
                "r_eff_over_r0": 2.0,
                "opening_strength_fraction": 1.0,
                "crack_extension_m": 5.0e-5,
                "signed_tau_over_sigma_by_system": [0.20, -0.10],
            },
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def test_signed_drive_family_interpolates_and_fails_closed(tmp_path):
    family = StateResolvedSignedDriveFamily.from_json(_write_drive_family(tmp_path))
    value = family.resolve(
        r_eff_over_r0=1.5,
        opening_strength_fraction=0.5,
        crack_extension_m=2.5e-5,
    )
    assert np.allclose(value, [0.15, -0.075])
    with pytest.raises(RuntimeError, match="outside"):
        family.resolve(
            r_eff_over_r0=2.5,
            opening_strength_fraction=0.5,
            crack_extension_m=2.5e-5,
        )


def test_drive_family_requires_same_states_as_kernel(tmp_path):
    family = StateResolvedSignedDriveFamily.from_json(_write_drive_family(tmp_path))
    kernel = SimpleNamespace(
        n_systems=2,
        states=[
            SimpleNamespace(state_id="low", coordinates=np.array([1.0, 0.0, 0.0])),
            SimpleNamespace(
                state_id="high", coordinates=np.array([2.0, 1.0, 5.0e-5])
            ),
        ],
    )
    family.validate_against_kernel_family(kernel)
    kernel.states = kernel.states[:1]
    with pytest.raises(ValueError, match="identical state IDs"):
        family.validate_against_kernel_family(kernel)


def _result(init, final):
    return {
        "status": "complete",
        "K_init_MPa_sqrt_m": float(init),
        "K_final_MPa_sqrt_m": float(final),
        "R_rise_MPa_sqrt_m": float(final - init),
        "R_rise_fraction": float((final - init) / init),
    }


def test_dbtt_score_targets_temperature_and_shielding_history():
    results = {
        ("full", 300.0): _result(10.0, 11.0),
        ("full", 700.0): _result(11.0, 14.0),
        ("full", 900.0): _result(12.0, 18.0),
        ("full", 1200.0): _result(13.0, 22.0),
        ("plasticity_off", 300.0): _result(10.0, 10.2),
        ("plasticity_off", 1200.0): _result(10.5, 10.7),
        ("shielding_off", 300.0): _result(10.0, 10.5),
        ("shielding_off", 1200.0): _result(11.0, 13.0),
        ("backstress_off", 300.0): _result(10.0, 11.0),
        ("backstress_off", 1200.0): _result(12.0, 24.0),
    }
    score = score_dbtt(results)
    assert score["strict_reduced_pass"]
    assert score["full_endpoint_ratio"] >= 1.5
    assert score["shielding_fraction_of_temperature_rise"] >= 0.5


def test_weakt_score_requires_weak_nonzero_r_curve():
    results = {}
    for T, init, final in (
        (300.0, 15.0, 16.5),
        (700.0, 15.2, 16.8),
        (900.0, 14.9, 16.4),
        (1200.0, 15.1, 16.7),
    ):
        results[("full", T)] = _result(init, final)
    for T in (300.0, 1200.0):
        results[("plasticity_off", T)] = _result(15.0, 15.3)
        results[("shielding_off", T)] = _result(15.0, 15.8)
        results[("backstress_off", T)] = _result(15.0, 17.0)
    score = score_weakt(results)
    assert score["strict_reduced_pass"]
    assert score["minimum_R_rise_fraction"] >= 0.05
    assert score["full_final_temperature_span_ratio"] <= 1.20


def test_ceramic_reference_is_frozen_flat_control():
    results = {
        ("full", T): _result(12.0, 12.3)
        for T in DEFAULT_TEMPERATURES_K
    }
    score = score_ceramic_reference(results)
    assert score["frozen_reference_pass"]
