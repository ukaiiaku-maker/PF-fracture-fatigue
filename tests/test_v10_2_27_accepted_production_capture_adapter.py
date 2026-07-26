from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
)
from arrhenius_fracture.physical_fem_capture_v10212 import CaptureRequest
from arrhenius_fracture.physical_fem_capture_v10213 import PhysicalFEMCapture

ROOT = Path(__file__).resolve().parents[1]
STATE_TABLE = ROOT / "scripts" / "write_v10_2_27_production_capture_state_table.py"
SEED_CHECK = ROOT / "scripts" / "check_v10_2_27_capture_seed_family.py"
RUNNER = ROOT / "scripts" / "run_v10_2_27_accepted_production_kernel_capture.sh"


def _write_configuration(path: Path, **changes) -> MechanicalKernelConfiguration:
    configuration = MechanicalKernelConfiguration(**changes)
    path.write_text(
        json.dumps(configuration.canonical_payload(), indent=2, sort_keys=True) + "\n"
    )
    return configuration


def test_state_table_uses_regular_anchors_and_exact_final_coverage(tmp_path: Path):
    config = tmp_path / "mechanical_configuration.json"
    _write_configuration(config)
    output = tmp_path / "states.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(STATE_TABLE),
            "--mechanical-config",
            str(config),
            "--required-max-extension-um",
            "1173.1",
            "--temperature-K",
            "700",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    extensions_um = [
        1.0e6 * float(row["cumulative_crack_path_extension_m"]) for row in rows
    ]
    assert extensions_um == pytest.approx(
        [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1175.0],
        rel=0.0,
        abs=1.0e-9,
    )
    tolerance_um = 1.0e6 * float(rows[0]["extension_tolerance_m"])
    assert 18.0 < tolerance_um < 20.0
    assert all(
        right - left > tolerance_um
        for left, right in zip(extensions_um[:-1], extensions_um[1:])
    )


def test_state_table_rejects_anchor_gap_smaller_than_maximum_event(tmp_path: Path):
    config = tmp_path / "mechanical_configuration.json"
    _write_configuration(config, atlas_anchor_spacing_m=10.0e-6)
    completed = subprocess.run(
        [
            sys.executable,
            str(STATE_TABLE),
            "--mechanical-config",
            str(config),
            "--required-max-extension-um",
            "100",
            "--temperature-K",
            "700",
            "--output",
            str(tmp_path / "states.csv"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "could cross more than one capture anchor" in (
        completed.stdout + completed.stderr
    )


def test_extension_capture_uses_first_crossing_not_symmetric_nearest(tmp_path: Path):
    request = CaptureRequest(
        state_id="E0000200",
        temperature_K=700.0,
        r_eff_over_r0=1.0,
        opening_strength_fraction=0.5,
        crack_extension_m=200.0e-6,
        r_tolerance=1.0e30,
        opening_tolerance=1.0,
        extension_tolerance_m=20.0e-6,
        interaction_ell_m=2.0e-6,
    ).validate()
    capture = PhysicalFEMCapture([request], tmp_path / "capture")
    before = capture._matching_request(700.0, {"crack_extension_m": 190.0e-6})
    crossed = capture._matching_request(700.0, {"crack_extension_m": 205.0e-6})
    missed = capture._matching_request(700.0, {"crack_extension_m": 225.0e-6})
    assert before is None
    assert crossed is request
    assert missed is None


def _write_seed_family(path: Path, maximum_um: float, *, authorized: bool = True) -> None:
    payload = {
        "schema": "v10.2.14_active_only_real_signed_2d_shielding_atlas",
        "production_parameterization_allowed": authorized,
        "campaign_parameterization_allowed": authorized,
        "active_kernel_mechanically_measured": True,
        "candidate_independent": True,
        "same_kernel_family_for_monotonic_and_fatigue": True,
        "frozen_geometry_load_invariance_passed": True,
        "normalization_is_mechanically_derived": True,
        "positive_and_negative_perturbations": True,
        "multi_amplitude_validation_passed": True,
        "wake_shielding_supported": False,
        "wake_kernel_forced_zero": True,
        "wake_kernel_mechanically_measured": False,
        "states": [
            {"state_id": "E0", "crack_extension_m": 0.0},
            {"state_id": "E1", "crack_extension_m": maximum_um * 1.0e-6},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_seed_family_checker_requires_authorization_and_coverage(tmp_path: Path):
    family = tmp_path / "family.json"
    _write_seed_family(family, 1200.0)
    completed = subprocess.run(
        [
            sys.executable,
            str(SEED_CHECK),
            "--family",
            str(family),
            "--required-path-extension-um",
            "1198",
            "--output",
            str(tmp_path / "audit.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["passed"] is True
    assert audit["maximum_path_extension_um"] == 1200.0
    assert audit["observed_authorization_gates"][
        "production_parameterization_allowed"
    ] is True

    _write_seed_family(family, 1190.0, authorized=False)
    failed = subprocess.run(
        [
            sys.executable,
            str(SEED_CHECK),
            "--family",
            str(family),
            "--required-path-extension-um",
            "1198",
            "--output",
            str(tmp_path / "failed_audit.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    combined = failed.stdout + failed.stderr
    assert "production_parameterization_allowed" in combined
    assert "required_path_extension_coverage" in combined


def test_capture_runner_preserves_exact_production_contract():
    text = RUNNER.read_text()
    required = (
        "KERNEL_CAPTURE_SEED_FAMILY",
        "KERNEL_CAPTURE_PARAMETER_OPTION",
        "KERNEL_CAPTURE_HAZARD_SEED",
        "check_v10_2_27_capture_seed_family.py",
        "sharp_front_v10_2_13_capture",
        "CLEAVAGE_HAZARD_MODE=exponential",
        "CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled",
        "--front-state-model moving_pz",
        "--tip-kinetics-mode moving_velocity",
        "--tip-plasticity",
        "--active-shielding",
        "--signed-active-shielding",
        "--no-wake-shielding",
        "PERSISTENT_SOURCE_MIN_WIDTH_UM",
    )
    for token in required:
        assert token in text
    assert "CLEAVAGE_HAZARD_MODE=deterministic" not in text
    assert "CLEAVAGE_EVENT_LENGTH_MODE=fixed" not in text
    assert "--no-tip-plasticity" not in text
    assert "--no-active-shielding" not in text
