from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
)
from arrhenius_fracture.kernel_registry_v10227 import (
    family_physics_fingerprint,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def test_kernel_fingerprint_excludes_temperature_when_mechanics_are_fixed():
    first = MechanicalKernelConfiguration(temperature_K=300.0)
    second = MechanicalKernelConfiguration(temperature_K=1300.0)
    assert first.fingerprint() == second.fingerprint()


def test_kernel_fingerprint_changes_with_orientation_topology_and_geometry():
    base = MechanicalKernelConfiguration()
    variants = (
        MechanicalKernelConfiguration(theta_deg=15.0),
        MechanicalKernelConfiguration(theta_deg=18.0),
        MechanicalKernelConfiguration(
            branching_mode="topology_cached", maximum_fronts=2
        ),
        MechanicalKernelConfiguration(process_zone_length_m=100.0e-6),
        MechanicalKernelConfiguration(process_zone_bins=40),
        MechanicalKernelConfiguration(mesh_nx=48),
        MechanicalKernelConfiguration(tip_h_fine_m=0.5e-6),
        MechanicalKernelConfiguration(interaction_length_m=4.0e-6),
    )
    for variant in variants:
        assert base.fingerprint() != variant.fingerprint()


def test_mechanical_configuration_contains_current_mesh_and_process_zone():
    cfg = MechanicalKernelConfiguration()
    assert cfg.process_zone_length_m == 50.0e-6
    assert cfg.process_zone_bins == 80
    assert cfg.mesh_nx == 36
    assert cfg.mesh_ny == 72
    assert cfg.tip_h_fine_m == 1.0e-6
    assert cfg.da_phys_m == 5.0e-6


def test_material_seed_and_target_do_not_change_mechanical_identity():
    base = MechanicalKernelConfiguration()
    payload = base.canonical_payload()
    payload["material_option"] = "not_a_mechanical_field"
    payload["hazard_seed"] = 928374
    payload["target_extension_um"] = 3000.0
    enriched = MechanicalKernelConfiguration.from_mapping(payload)
    assert enriched.fingerprint() == base.fingerprint()


def test_unknown_mechanical_key_fails_closed():
    payload = MechanicalKernelConfiguration().canonical_payload()
    payload["misspelled_mesh_setting"] = 7
    try:
        MechanicalKernelConfiguration.from_mapping(payload)
    except ValueError as exc:
        assert "unknown mechanical-configuration keys" in str(exc)
    else:
        raise AssertionError("unknown mechanical key was silently accepted")


def test_path_independent_family_physics_fingerprint(tmp_path: Path):
    common = {
        "schema": "v10.2.14_active_only_real_signed_2d_shielding_atlas",
        "states": [
            {"state_id": "E0", "crack_extension_m": 0.0, "values": [1.0, 2.0]},
            {"state_id": "E1", "crack_extension_m": 1.0e-3, "values": [3.0, 4.0]},
        ],
        "active_kernel_mechanically_measured": True,
    }
    first = dict(common, response="/old/machine/run/E0.csv", source_path="/old/family.json")
    second = dict(common, response="/new/cache/run/E0.csv", source_path="/new/family.json")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first))
    second_path.write_text(json.dumps(second))
    assert family_physics_fingerprint(first_path) == family_physics_fingerprint(second_path)


def test_tracked_registry_does_not_require_archives():
    registry = load_registry(ROOT / "artifacts" / "v10_2_27_kernel_registry.json")
    assert registry["recipes"] == []
    assert registry["policy"]["portable_archives_are_required_inputs"] is False
    assert registry["policy"]["default_resolution"] == (
        "recalculate_from_current_mechanical_configuration"
    )


def test_official_runners_resolve_instead_of_hardcoding_run_paths():
    for relative in (
        "scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves_validated.sh",
        "scripts/run_v10_2_27_replace_weakT_ceramic_1000um.sh",
    ):
        text = (ROOT / relative).read_text()
        assert "resolve_v10_2_27_kernel_for_runner.sh" in text
        assert "FAMILY_JSON=${FAMILY_JSON:-$ROOT/runs/" not in text


def test_default_builder_recalculates_capture_and_load_invariance():
    builder = (
        ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"
    ).read_text()
    assert "capture_v10_2_27_kernel_states_for_configuration.py" in builder
    assert "RECALCULATE frozen FEM states" in builder
    assert "RECALCULATE load invariance" in builder
    assert "KERNEL_CAPTURE_COMMAND:?" not in builder


def test_capture_uses_current_registry_geometry_not_historical_archive():
    capture = (
        ROOT / "scripts" / "capture_v10_2_27_kernel_states_for_configuration.py"
    ).read_text()
    assert "v10_2_27_v913_four_class_paper_registry.csv" in capture
    assert '"--mpz-length-um"' in capture
    assert '"--mpz-n-bins"' in capture
    assert '"--no-tip-plasticity"' in capture
    assert '"CLEAVAGE_HAZARD_MODE": "deterministic"' in capture
    assert "internally_generated_mechanics_only_manifest" in capture
    assert "trajectory-option" not in capture


def test_optional_archives_require_exact_configuration_provenance():
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_from_current_mechanics.py"
    ).read_text()
    assert "kernel_capture_manifest.json" in text
    assert "Legacy archives cannot be assumed compatible" in text
    assert "expected_configuration_fingerprint" in text


def test_current_registry_geometry_is_50um_80bins():
    import csv

    path = (
        ROOT
        / "arrhenius_fracture"
        / "data"
        / "materials"
        / "v10_2_27_v913_four_class_paper_registry.csv"
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {float(row["L_pz_um_recommended"]) for row in rows} == {50.0}
    assert {int(float(row["n_bins_recommended"])) for row in rows} == {80}


def test_resolver_rebuilds_when_requested_coverage_is_longer():
    text = (ROOT / "scripts" / "ensure_v10_2_27_signed_kernel.py").read_text()
    assert "cached coverage is too short" in text
    assert "_clear_generated_cache" in text
    assert '"resolution": "recalculated"' in text


def test_branching_cannot_fall_back_to_single_front_atlas(tmp_path: Path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "v10.2.27_kernel_registry_v1",
                "entries": [],
                "recipes": [],
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ensure_v10_2_27_signed_kernel.py"),
            "--theta-deg",
            "30",
            "--target-extension-um",
            "100",
            "--branching-mode",
            "topology_cached",
            "--maximum-fronts",
            "2",
            "--mode",
            "build",
            "--tracked-registry",
            str(registry),
            "--cache-root",
            str(tmp_path / "cache"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "no branch-aware kernel provider" in (completed.stdout + completed.stderr)
