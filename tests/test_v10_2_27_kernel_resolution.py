from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
    endpoint_resolving_tip_h_fine_m,
)
from arrhenius_fracture.kernel_registry_v10227 import (
    family_physics_fingerprint,
    load_registry,
)
from arrhenius_fracture.kernel_resolver_v10227 import (
    _configuration,
    build_parser,
)

ROOT = Path(__file__).resolve().parents[1]


def _resolver_args(*extra: str):
    return build_parser().parse_args(
        ["--theta-deg", "30", "--target-extension-um", "1000", *extra]
    )


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
        MechanicalKernelConfiguration(tip_h_fine_m=0.5 * base.tip_h_fine_m),
        MechanicalKernelConfiguration(
            measurement_tip_h_fine_m=0.5 * base.measurement_tip_h_fine_m
        ),
        MechanicalKernelConfiguration(interaction_length_m=4.0e-6),
        MechanicalKernelConfiguration(specimen_length_x_m=3.0e-3),
    )
    for variant in variants:
        assert base.fingerprint() != variant.fingerprint()


def test_mechanical_configuration_separates_production_and_measurement_meshes():
    cfg = MechanicalKernelConfiguration()
    assert cfg.process_zone_length_m == 50.0e-6
    assert cfg.process_zone_bins == 80
    assert cfg.mesh_nx == 36
    assert cfg.mesh_ny == 72
    assert cfg.tip_h_fine_m == 1.0e-6
    assert cfg.measurement_tip_h_fine_m == endpoint_resolving_tip_h_fine_m(
        50.0e-6, 80
    )
    assert cfg.measurement_tip_h_fine_m < cfg.tip_h_fine_m
    assert cfg.da_phys_m == 5.0e-6
    assert cfg.specimen_length_x_m == 2.0e-3
    assert cfg.specimen_length_y_m == 4.0e-3
    assert cfg.initial_crack_length_m == 0.5e-3


def test_coarse_measurement_mesh_is_rejected_without_restricting_production_mesh():
    accepted = MechanicalKernelConfiguration(tip_h_fine_m=1.0e-6)
    assert accepted.tip_h_fine_m == 1.0e-6
    try:
        MechanicalKernelConfiguration(measurement_tip_h_fine_m=1.0e-6)
    except ValueError as exc:
        assert "measurement_tip_h_fine_m is too coarse" in str(exc)
    else:
        raise AssertionError("under-resolved measurement endpoint was accepted")


def test_generated_active_grid_derives_measurement_spacing_only():
    cfg = _configuration(
        _resolver_args(
            "--process-zone-length-um", "100",
            "--process-zone-bins", "100",
        )
    )
    assert cfg.tip_h_fine_m == 1.0e-6
    assert cfg.measurement_tip_h_fine_m == endpoint_resolving_tip_h_fine_m(
        100.0e-6, 100
    )


def test_material_seed_and_target_fields_do_not_directly_change_identity():
    base = MechanicalKernelConfiguration()
    payload = base.canonical_payload()
    payload["material_option"] = "not_a_mechanical_field"
    payload["hazard_seed"] = 928374
    payload["target_extension_um"] = 3000.0
    enriched = MechanicalKernelConfiguration.from_mapping(payload)
    assert enriched.fingerprint() == base.fingerprint()


def test_target_extension_reuses_domain_when_it_fits_and_enlarges_when_needed():
    short = _configuration(_resolver_args())
    still_fits = _configuration(_resolver_args("--target-extension-um", "1200"))
    long = _configuration(_resolver_args("--target-extension-um", "3000"))
    assert short.fingerprint() == still_fits.fingerprint()
    assert short.specimen_length_x_m == 2.0e-3
    assert long.specimen_length_x_m > short.specimen_length_x_m
    assert long.fingerprint() != short.fingerprint()


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
    first = dict(common, response="/old/run/E0.csv", source_path="/old/family.json")
    second = dict(common, response="/new/run/E0.csv", source_path="/new/family.json")
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


def test_legacy_theta30_archives_and_restore_scripts_are_removed():
    artifact_root = ROOT / "artifacts" / "v10_2_27_theta30_frontfix"
    assert not list(artifact_root.glob("*.zip"))
    for name in (
        "canonicalize_v10_2_27_theta30_snapshot_archive.py",
        "restore_v10_2_27_theta30_snapshot_archive.py",
        "canonicalize_v10_2_27_theta30_load_invariance_archive.py",
        "restore_v10_2_27_theta30_load_invariance_archive.py",
    ):
        assert not (ROOT / "scripts" / name).exists()


def test_official_runners_resolve_instead_of_hardcoding_run_paths():
    for relative in (
        "scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves_validated.sh",
        "scripts/run_v10_2_27_replace_weakT_ceramic_1000um.sh",
    ):
        text = (ROOT / relative).read_text()
        assert "resolve_v10_2_27_kernel_for_runner.sh" in text
        assert "FAMILY_JSON=${FAMILY_JSON:-$ROOT/runs/" not in text


def test_runner_resolver_ignores_inherited_family_by_default():
    text = (ROOT / "scripts" / "resolve_v10_2_27_kernel_for_runner.sh").read_text()
    assert "Ignoring inherited FAMILY_JSON" in text
    assert "KERNEL_USE_FAMILY_OVERRIDE" in text
    assert "MECHANICAL_PROFILE_OVERRIDE" in text


def test_default_builder_requires_accepted_production_capture_then_recalculates_kernel():
    builder = (
        ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"
    ).read_text()
    assert "automatic mechanics-only kernel capture is disabled" in builder
    assert "KERNEL_CAPTURE_COMMAND" in builder
    assert "capture_v10_2_27_kernel_states_for_configuration.py" not in builder
    assert "CAPTURE accepted production states with registered command" in builder
    assert "RECALCULATE load invariance endpoints" in builder
    assert '--minimum-station-spacing-m "$PZ_LENGTH_M"' in builder


def test_capture_hook_separates_trajectory_physics_from_measurement_mesh():
    base = (
        ROOT / "arrhenius_fracture" / "physical_fem_capture_v10212.py"
    ).read_text()
    entry = (
        ROOT / "arrhenius_fracture" / "sharp_front_v10_2_13_capture.py"
    ).read_text()
    assert "measurement_mesh_config" in base
    assert "reconstruct_frozen_measurement_state" in base
    assert "_engine_kinetic_state_digest" in base
    assert "production_engine_state_bitwise_unchanged" in base
    assert "measurement_reconstruction_called_mpz_advance" in base
    assert "--atlas-measurement-tip-h-fine" in entry
    assert "production_kinetics=unchanged" in entry


def test_optional_archives_require_exact_configuration_and_capture_provenance():
    text = (
        ROOT / "scripts" / "build_v10_2_27_kernel_from_current_mechanics.py"
    ).read_text()
    assert "kernel_capture_manifest.json" in text
    assert "Legacy archives cannot be assumed compatible" in text
    assert "expected_configuration_fingerprint" in text
    assert "production_engine_state_bitwise_unchanged" in text
    assert "measurement_reconstruction_called_mpz_advance" in text


def test_current_registry_geometry_is_50um_80bins():
    import csv

    path = (
        ROOT / "arrhenius_fracture" / "data" / "materials"
        / "v10_2_27_v913_four_class_paper_registry.csv"
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {float(row["L_pz_um_recommended"]) for row in rows} == {50.0}
    assert {int(float(row["n_bins_recommended"])) for row in rows} == {80}


def test_resolver_rebuilds_when_requested_coverage_is_longer():
    text = (ROOT / "arrhenius_fracture" / "kernel_resolver_v10227.py").read_text()
    assert "cached coverage is too short" in text
    assert "_clear_generated_cache" in text
    assert '"resolution": "recalculated"' in text


def test_replacement_runner_requires_matching_retained_kernel_provenance():
    text = (
        ROOT / "scripts" / "run_v10_2_27_replace_weakT_ceramic_1000um.sh"
    ).read_text()
    assert "check_v10_2_27_retained_kernel_compatibility.py" in text


def test_retained_case_without_fingerprint_is_rejected(tmp_path: Path):
    family = tmp_path / "family.json"
    family.write_text(json.dumps({"mechanical_configuration_fingerprint": "new-config"}))
    option = "retained"
    case = tmp_path / option / "T300K_th30_seed10"
    case.mkdir(parents=True)
    (case / "v10_2_27_paper_four_class_parameter_transfer.json").write_text(
        json.dumps({"selected_option": option})
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_v10_2_27_retained_kernel_compatibility.py"),
            "--family", str(family),
            "--outroot", str(tmp_path),
            "--options", option,
            "--temperatures", "300",
            "--theta-deg", "30",
            "--base-seed", "10",
            "--seed-option-stride", "100",
            "--seed-temperature-stride", "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "RETAINED KERNEL COMPATIBILITY FAILED" in (
        completed.stdout + completed.stderr
    )


def test_branching_cannot_fall_back_to_single_front_atlas(tmp_path: Path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "schema": "v10.2.27_kernel_registry_v1",
        "entries": [],
        "recipes": [],
    }))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ensure_v10_2_27_signed_kernel.py"),
            "--theta-deg", "30",
            "--target-extension-um", "100",
            "--branching-mode", "topology_cached",
            "--maximum-fronts", "2",
            "--mode", "build",
            "--tracked-registry", str(registry),
            "--cache-root", str(tmp_path / "cache"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "no branch-aware kernel provider" in (completed.stdout + completed.stderr)
