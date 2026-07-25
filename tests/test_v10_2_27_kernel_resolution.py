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
    select_recipe,
)

ROOT = Path(__file__).resolve().parents[1]


def test_kernel_fingerprint_excludes_temperature_when_mechanics_are_fixed():
    first = MechanicalKernelConfiguration(temperature_K=300.0)
    second = MechanicalKernelConfiguration(temperature_K=1300.0)
    assert first.fingerprint() == second.fingerprint()


def test_kernel_fingerprint_changes_with_orientation_and_topology():
    base = MechanicalKernelConfiguration()
    rotated = MechanicalKernelConfiguration(theta_deg=45.0)
    branched = MechanicalKernelConfiguration(
        branching_mode="topology_cached", maximum_fronts=2
    )
    assert base.fingerprint() != rotated.fingerprint()
    assert base.fingerprint() != branched.fingerprint()


def test_material_or_seed_fields_do_not_change_mechanical_identity():
    base = MechanicalKernelConfiguration()
    payload = base.canonical_payload()
    payload["material_option"] = "not_a_mechanical_field"
    payload["hazard_seed"] = 928374
    payload["target_extension_um"] = 2500.0
    enriched = MechanicalKernelConfiguration.from_mapping(payload)
    assert enriched.fingerprint() == base.fingerprint()


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


def test_default_theta30_recipe_is_registered():
    registry = load_registry(ROOT / "artifacts" / "v10_2_27_kernel_registry.json")
    configuration = MechanicalKernelConfiguration().canonical_payload()
    recipe = select_recipe(registry, configuration)
    assert recipe is not None
    assert recipe["builder"] == "portable_mechanics_artifacts"
    assert recipe["normalization_policy"] == "derive_v10.2.12_from_snapshot_engine_config"


def test_official_runners_resolve_kernels_instead_of_hardcoding_run_paths():
    for relative in (
        "scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves_validated.sh",
        "scripts/run_v10_2_27_replace_weakT_ceramic_1000um.sh",
    ):
        text = (ROOT / relative).read_text()
        assert "resolve_v10_2_27_kernel_for_runner.sh" in text
        assert "FAMILY_JSON=${FAMILY_JSON:-$ROOT/runs/" not in text


def test_portable_builder_derives_normalization_from_snapshot_engine_configuration():
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_from_mechanics_artifacts.py"
    ).read_text()
    assert "derive_mechanical_normalization" in text
    assert "portable_path_relocation" in text
    assert "--normalization" in text


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
