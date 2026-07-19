import csv
import json
from pathlib import Path

import pytest

from arrhenius_fracture.physical_artifact_workflow_v10210 import (
    MechanicalStateRequest,
    inspect_raw_artifacts,
    readiness_report,
    validate_review_approval,
    write_collection_plan,
)


def _engine(path: Path):
    path.write_text(
        json.dumps(
            {
                "front_config": {"sigma_cap": 3.0e10},
                "mpz_config": {"n_bins": 4},
                "tip_config": {},
                "anisotropic_config": {},
                "transport_mode": "validated_scalar",
            }
        )
    )
    return path


def _states(engine: Path):
    result = []
    index = 0
    for r in (0.8, 1.2):
        for opening in (0.5, 0.75, 1.0):
            for extension in (0.0, 1.0e-5):
                result.append(
                    MechanicalStateRequest(
                        state_id=f"S{index:02d}",
                        r_eff_over_r0=r,
                        opening_strength_fraction=opening,
                        crack_extension_m=extension,
                        engine_template=engine,
                    )
                )
                index += 1
    return result


def test_collection_plan_is_header_only_and_fail_closed(tmp_path):
    engine = _engine(tmp_path / "engine.json")
    out = tmp_path / "plan"
    payload = write_collection_plan(
        _states(engine),
        outroot=out,
        n_systems=2,
        active_bins=2,
        wake_bins=1,
        perturbation_magnitudes=[0.25, 0.5],
    )
    assert payload["state_coverage"]["coverage_passed"] is True
    assert payload["physical_fem_responses_generated"] is False
    assert payload["automatic_authorization"] is False
    assert len((out / "signed_interaction_integral_responses.csv").read_text().splitlines()) == 1
    assert len((out / "tensor_probe_responses.csv").read_text().splitlines()) == 1
    with (out / "mechanical_state_jobs.csv").open(newline="") as handle:
        jobs = list(csv.DictReader(handle))
    assert jobs
    assert {row["status"] for row in jobs} == {"pending_physical_fem"}


def _write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_raw_preflight_requires_matching_states_and_mechanical_normalization(tmp_path):
    signed = tmp_path / "signed.csv"
    tensor = tmp_path / "tensor.csv"
    normalization = tmp_path / "normalization.json"
    signed_row = {
        "state_id": "S0",
        "r_eff_over_r0": 1.0,
        "opening_strength_fraction": 0.5,
        "crack_extension_m": 0.0,
        "region": "active",
        "system": 0,
        "bin": 0,
        "x_m": 1.0e-6,
        "burgers_sign": 1,
        "delta_signed_line_content": 0.25,
        "K_I_base_Pa_sqrt_m": 1.0,
        "K_I_perturbed_Pa_sqrt_m": 0.9,
        "K_II_base_Pa_sqrt_m": 0.0,
        "K_II_perturbed_Pa_sqrt_m": 0.0,
    }
    tensor_row = {
        "state_id": "S0",
        "r_eff_over_r0": 1.0,
        "opening_strength_fraction": 0.5,
        "crack_extension_m": 0.0,
        "system": 0,
        "sigma_local_Pa": 1.0e9,
        "tau_signed_Pa": 2.0e8,
        "probe_reliable": True,
    }
    _write_csv(signed, list(signed_row), [signed_row])
    _write_csv(tensor, list(tensor_row), [tensor_row])
    normalization.write_text(
        json.dumps(
            {
                "normalization_source": "2d_unit_slip_to_line_content",
                "activation_to_line_content_by_system": [1.0],
                "source_capacity_bounds_per_system": [[1.0, 10.0]],
                "fitted_to_toughness_or_fatigue": False,
            }
        )
    )
    report = inspect_raw_artifacts(
        signed_responses=signed,
        tensor_responses=tensor,
        normalization=normalization,
    )
    assert report["raw_inputs_present"] is True
    assert report["production_parameterization_allowed"] is False


def test_review_approval_requires_every_independent_gate(tmp_path):
    approval = tmp_path / "approval.json"
    payload = {
        "schema": "v10.2.10_independent_mechanics_review",
        "reviewer": "reviewer",
        "reviewed_utc": "2026-07-19T00:00:00Z",
        "signed_linearity_passed": True,
        "multi_amplitude_linearity_passed": True,
        "interaction_integral_contour_stability_passed": True,
        "mesh_and_ribbon_convergence_passed": True,
        "state_envelope_coverage_passed": True,
        "source_normalization_reviewed": True,
        "tensor_probe_repeatability_passed": True,
        "replay_gate_passed": False,
    }
    approval.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete checks"):
        validate_review_approval(approval)
    payload["replay_gate_passed"] = True
    approval.write_text(json.dumps(payload))
    assert validate_review_approval(approval)["reviewer"] == "reviewer"


def test_readiness_requires_both_authorizations_and_state_match(tmp_path):
    engine = _engine(tmp_path / "engine.json")
    kernel = tmp_path / "kernel.json"
    drive = tmp_path / "drive.json"
    kernel.write_text(
        json.dumps(
            {
                "production_parameterization_allowed": True,
                "states": [{"state_id": "S0"}],
            }
        )
    )
    drive.write_text(
        json.dumps(
            {
                "production_parameterization_allowed": True,
                "states": [{"state_id": "S0"}],
            }
        )
    )
    assert readiness_report(
        kernel_family=kernel,
        drive_family=drive,
        engine_template=engine,
    )["ready_for_stage_1"] is True
    drive.write_text(
        json.dumps(
            {
                "production_parameterization_allowed": True,
                "states": [{"state_id": "S1"}],
            }
        )
    )
    assert readiness_report(
        kernel_family=kernel,
        drive_family=drive,
        engine_template=engine,
    )["ready_for_stage_1"] is False
