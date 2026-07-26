from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
)
from arrhenius_fracture.kernel_registry_v10227 import family_physics_fingerprint
from arrhenius_fracture.kernel_resolver_v10227 import (
    _validate_promotion_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT / "scripts" / "compare_v10_2_27_kernel_families.py"
BUILDER_PATH = ROOT / "scripts" / "build_v10_2_27_kernel_from_current_mechanics.py"
SPEC = importlib.util.spec_from_file_location("v10227_current_builder_test", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _family(path: Path, *, scale: float = 1.0, source_path: str = "/old/run") -> None:
    payload = {
        "schema": "v10.2.14_active_only_real_signed_2d_shielding_atlas",
        "production_parameterization_allowed": True,
        "campaign_parameterization_allowed": True,
        "active_kernel_mechanically_measured": True,
        "wake_kernel_forced_zero": True,
        "wake_kernel_mechanically_measured": False,
        "wake_shielding_supported": False,
        "active_x_m": [0.25e-6, 0.75e-6],
        "activation_to_line_content_by_system": [0.91, 0.91],
        "source_capacity_bounds_per_system": [[100.0, 200.0], [100.0, 200.0]],
        "source_path": source_path,
        "states": [
            {
                "state_id": "E0000000",
                "crack_extension_m": 0.0,
                "active_kernel_I_Pa_sqrt_m_per_signed_line": [
                    1000.0 * scale,
                    -500.0 * scale,
                ],
                "active_kernel_II_Pa_sqrt_m_per_signed_line": [
                    100.0 * scale,
                    50.0 * scale,
                ],
                "response": source_path + "/E0.csv",
            },
            {
                "state_id": "E0000200",
                "crack_extension_m": 200.0e-6,
                "active_kernel_I_Pa_sqrt_m_per_signed_line": [
                    900.0 * scale,
                    -450.0 * scale,
                ],
                "active_kernel_II_Pa_sqrt_m_per_signed_line": [
                    90.0 * scale,
                    45.0 * scale,
                ],
                "response": source_path + "/E200.csv",
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_compare(previous: Path, current: Path, output: Path):
    return subprocess.run(
        [
            sys.executable,
            str(COMPARATOR),
            "--previous",
            str(previous),
            "--current",
            str(current),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_comparator_ignores_paths_but_records_file_and_physics_hashes(tmp_path: Path):
    previous = tmp_path / "iteration_00" / "family.json"
    current = tmp_path / "iteration_01" / "family.json"
    previous.parent.mkdir()
    current.parent.mkdir()
    _family(previous, source_path="/old/run")
    _family(current, source_path="/new/run")

    completed = _run_compare(previous, current, tmp_path / "comparison.json")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["converged"] is True
    assert comparison["same_file_sha256"] is False
    assert comparison["same_physics_fingerprint"] is True
    assert comparison["previous_family_label"] == "iteration_00/family.json"
    assert comparison["current_family_label"] == "iteration_01/family.json"
    assert len(comparison["current_family_sha256"]) == 64
    assert len(comparison["current_family_physics_fingerprint"]) == 64


def test_comparator_rejects_material_kernel_change(tmp_path: Path):
    previous = tmp_path / "previous.json"
    current = tmp_path / "current.json"
    _family(previous, scale=1.0)
    _family(current, scale=1.25)

    completed = _run_compare(previous, current, tmp_path / "comparison.json")
    assert completed.returncode == 3
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["converged"] is False
    assert "active_kernel_change" in comparison["failures"]


def _selection(configuration: MechanicalKernelConfiguration, candidate: Path) -> dict:
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    candidate_physics = family_physics_fingerprint(candidate)
    comparison = {
        "schema": "v10.2.27_kernel_self_consistency_comparison_v1",
        "previous_family_sha256": "a" * 64,
        "previous_family_physics_fingerprint": "b" * 64,
        "current_family_sha256": candidate_sha,
        "current_family_physics_fingerprint": candidate_physics,
        "converged": True,
        "failures": [],
    }
    return {
        "schema": "v10.2.27_kernel_self_consistency_selection_v2",
        "mechanical_configuration_fingerprint": configuration.fingerprint(),
        "initial_bootstrap_family_sha256": "c" * 64,
        "converged": True,
        "converged_iteration": 1,
        "minimum_target_family_passes": 2,
        "converged_candidate_family_sha256": candidate_sha,
        "converged_candidate_family_physics_fingerprint": candidate_physics,
        "comparisons": [comparison],
    }


def test_promotion_gate_requires_portable_converged_selection(tmp_path: Path):
    configuration = MechanicalKernelConfiguration()
    candidate = tmp_path / "family.json"
    _family(candidate)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    with pytest.raises(ValueError, match="single capture/build pass"):
        BUILDER._validate_self_consistency_selection(
            snapshots,
            configuration,
            allow_unconverged_capture=False,
        )

    payload = _selection(configuration, candidate)
    (snapshots / "kernel_self_consistency_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    result = BUILDER._validate_self_consistency_selection(
        snapshots,
        configuration,
        allow_unconverged_capture=False,
    )
    assert result["validated"] is True
    assert result["selection"][
        "converged_candidate_family_physics_fingerprint"
    ] == family_physics_fingerprint(candidate)


def test_promotion_gate_rejects_candidate_physics_mismatch(tmp_path: Path):
    configuration = MechanicalKernelConfiguration()
    candidate = tmp_path / "family.json"
    _family(candidate)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    payload = _selection(configuration, candidate)
    payload["converged_candidate_family_physics_fingerprint"] = "d" * 64
    (snapshots / "kernel_self_consistency_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="candidate_physics_fingerprint"):
        BUILDER._validate_self_consistency_selection(
            snapshots,
            configuration,
            allow_unconverged_capture=False,
        )


def test_provisional_iteration_rejects_accidental_final_selection(tmp_path: Path):
    configuration = MechanicalKernelConfiguration()
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "kernel_self_consistency_selection.json").write_text("{}\n")
    with pytest.raises(ValueError, match="provisional iteration capture"):
        BUILDER._validate_self_consistency_selection(
            snapshots,
            configuration,
            allow_unconverged_capture=True,
        )


def test_resolver_rejects_cache_without_promotion_manifests(tmp_path: Path):
    candidate = tmp_path / "family.json"
    _family(candidate)
    audit = {
        "file_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "physics_fingerprint": family_physics_fingerprint(candidate),
    }
    with pytest.raises(ValueError, match="lacks fixed-point promotion manifests"):
        _validate_promotion_evidence(tmp_path, audit, "f" * 64)


def test_resolver_accepts_only_matching_promotion_manifests(tmp_path: Path):
    configuration = MechanicalKernelConfiguration()
    candidate = tmp_path / "family.json"
    _family(candidate)
    family_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    physics = family_physics_fingerprint(candidate)
    selection = _selection(configuration, candidate)
    build_manifest = {
        "schema": "v10.2.27_current_configuration_kernel_build_v5",
        "configuration_fingerprint": configuration.fingerprint(),
        "family_sha256": family_sha,
        "family_physics_fingerprint": physics,
        "production_parameterization_promotion_allowed": True,
        "self_consistency": {
            "validated": True,
            "selection": selection,
        },
    }
    consistency_manifest = {
        "schema": "v10.2.27_kernel_self_consistency_manifest_v3",
        "converged": True,
        "converged_iteration": 1,
        "canonical_family_sha256": family_sha,
        "canonical_family_physics_fingerprint": physics,
        "selection": selection,
    }
    (tmp_path / "kernel_build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n"
    )
    (tmp_path / "kernel_self_consistency_manifest.json").write_text(
        json.dumps(consistency_manifest, indent=2, sort_keys=True) + "\n"
    )
    audit = {"file_sha256": family_sha, "physics_fingerprint": physics}
    result = _validate_promotion_evidence(
        tmp_path, audit, configuration.fingerprint()
    )
    assert result["production_parameterization_promotion_allowed"] is True
    assert result["converged_iteration"] == 1

    consistency_manifest["canonical_family_physics_fingerprint"] = "e" * 64
    (tmp_path / "kernel_self_consistency_manifest.json").write_text(
        json.dumps(consistency_manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="canonical_physics_fingerprint"):
        _validate_promotion_evidence(
            tmp_path, audit, configuration.fingerprint()
        )
