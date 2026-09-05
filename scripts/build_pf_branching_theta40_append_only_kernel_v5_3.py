#!/usr/bin/env python3
"""Build the authorized append-only theta-40 signed-kernel family to 745 um.

The historical 0--415 um state and load-invariance payloads are copied without
modification.  Only 420, 425, 600, and 745 um are newly measured.  The emitted
family opts into the v10.2.14 exact-prefix resolver so appending measurements
cannot perturb interpolation anywhere in the qualified historical domain.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from arrhenius_fracture.kernel_configuration_v10227 import load_configuration
from arrhenius_fracture.kernel_registry_v10227 import (
    family_physics_fingerprint,
    sha256_file,
    validate_family,
)
from arrhenius_fracture.prescribed_geometry_kernel_v10228 import (
    build_prescribed_geometry_snapshots,
)
from arrhenius_fracture.prescribed_geometry_numpy2_compat_v10228 import (
    install_numpy2_orientation_compat,
)
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)

SCHEMA = "pf_current_source_branching_theta40_append_only_kernel_v5_3"
OLD_LEVELS_UM = (0.0, 200.0, 400.0, 415.0)
NEW_LEVELS_UM = (420.0, 425.0, 600.0, 745.0)
ALL_LEVELS_UM = OLD_LEVELS_UM + NEW_LEVELS_UM
OLD_STATE_IDS = ("E0000000", "E0000200", "E0000400", "E0000415")
NEW_STATE_IDS = ("E0000420", "E0000425", "E0000600", "E0000745")
EXPECTED_OLD_FAMILY_SHA256 = (
    "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847"
)
EXPECTED_CONFIGURATION_SHA256 = (
    "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9"
)
EXPECTED_CONFIGURATION_FINGERPRINT = (
    "adb7754436a66542a38c17d671bc62639939d85075168a5db721b93b791e87d0"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.stdout:
        print(completed.stdout, end="", file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"append-only kernel command failed ({completed.returncode}): "
            + " ".join(command)
        )


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_and_verify(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copytree(source, destination, copy_function=shutil.copyfile)
    before = _tree_manifest(source)
    after = _tree_manifest(destination)
    if before != after:
        raise RuntimeError(f"byte identity failed while copying {source.name}")
    return {
        "state_id": source.name,
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "file_count": len(before),
        "tree_files": before,
        "byte_identical": True,
    }


def _combined_state_rows(
    snapshot_root: Path,
    old_manifest: dict[str, Any],
    new_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    source_rows = {
        str(row["state_id"]): row
        for manifest in (old_manifest, new_manifest)
        for row in manifest.get("states", [])
    }
    rows: list[dict[str, Any]] = []
    for state_id in OLD_STATE_IDS + NEW_STATE_IDS:
        path = snapshot_root / state_id / "snapshot.json"
        if state_id not in source_rows:
            raise RuntimeError(f"capture manifest is missing {state_id}")
        row = copy.deepcopy(source_rows[state_id])
        row["snapshot"] = str(path.resolve())
        rows.append(row)
    return rows


def _validate_prefix_state_rows(old_payload: dict[str, Any], new_payload: dict[str, Any]) -> None:
    old_rows = old_payload.get("states", [])
    new_rows = new_payload.get("states", [])[: len(old_rows)]
    if old_rows != new_rows:
        raise RuntimeError("assembled family changed at least one preserved 0--415 um state")


def _canonicalize_artifact_value(value: Any) -> Any:
    """Remove run-root and temporary-directory identity from embedded evidence."""
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_artifact_value(item)
            for key, item in value.items()
            if not str(key).lower().endswith("_sha256")
            and str(key).lower() != "sha256"
        }
    if isinstance(value, list):
        return [_canonicalize_artifact_value(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return f"<artifact>/{Path(value).name}"
    return value


def _canonical_digest(value: Any) -> str:
    payload = _canonicalize_artifact_value(value)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_family_provenance(
    payload: dict[str, Any],
    *,
    loads: Path,
) -> None:
    """Retain evidence while making the portable family byte reproducible."""
    stable_responses = []
    stable_loads = []
    stable_campaign_states = []
    extension_by_id = {
        str(row["state_id"]): float(row["crack_extension_m"])
        for row in payload.get("states", [])
    }
    for state_id in OLD_STATE_IDS + NEW_STATE_IDS:
        state_root = loads / state_id
        response = state_root / "active_station_responses_load_1.csv"
        audit_path = response.with_suffix(".audit.json")
        report_path = state_root / "frozen_geometry_load_invariance.json"
        audit = _canonicalize_artifact_value(_load(audit_path))
        report = _load(report_path)
        checks = report["checks"]
        audit_digest = _canonical_digest(audit)
        report_digest = _canonical_digest(report)
        response_digest = sha256_file(response)
        relative_response = f"load_invariance/{state_id}/{response.name}"
        relative_audit = f"load_invariance/{state_id}/{audit_path.name}"
        relative_report = f"load_invariance/{state_id}/{report_path.name}"
        stable_responses.append(
            {
                "state_id": state_id,
                "active_only": True,
                "path": relative_response,
                "sha256": response_digest,
                "audit_path": relative_audit,
                "audit_sha256": audit_digest,
                "audit": audit,
            }
        )
        stable_loads.append(
            {
                "parent_state_id": state_id,
                "path": relative_report,
                "scientific_payload_sha256": report_digest,
                "maximum_relative_load_variation": checks[
                    "maximum_relative_load_variation"
                ],
            }
        )
        stable_campaign_states.append(
            {
                "state_id": state_id,
                "cumulative_crack_path_extension_m": extension_by_id[state_id],
                "response": relative_response,
                "response_sha256": response_digest,
                "response_audit_sha256": audit_digest,
                "load_invariance_report": relative_report,
                "load_invariance_scientific_payload_sha256": report_digest,
                "maximum_relative_load_variation": checks[
                    "maximum_relative_load_variation"
                ],
                "maximum_within_load_relative_spread": checks[
                    "maximum_within_load_relative_spread"
                ],
            }
        )
    payload["physical_response_inputs"] = stable_responses
    payload["frozen_geometry_load_invariance_inputs"] = stable_loads
    campaign = dict(payload.get("campaign_promotion", {}))
    campaign["input_states"] = stable_campaign_states
    campaign["mechanics_normalization"] = (
        "portable_assembly/mechanics_normalization.json"
    )
    payload["campaign_promotion"] = campaign
    payload["normalization_artifact"] = (
        "portable_assembly/mechanics_normalization.json"
    )
    payload["normalization_path"] = "portable_assembly/mechanics_normalization.json"
    payload["unit_response_table"] = (
        "generated_transiently_from_measured_state_responses"
    )

    def absolute_values(value: Any, path: tuple[str, ...] = ()) -> list[str]:
        if isinstance(value, dict):
            return [
                found
                for key, item in value.items()
                for found in absolute_values(item, path + (str(key),))
            ]
        if isinstance(value, list):
            return [
                found
                for index, item in enumerate(value)
                for found in absolute_values(item, path + (str(index),))
            ]
        if isinstance(value, str) and Path(value).is_absolute():
            return [".".join(path)]
        return []

    remaining = absolute_values(payload)
    if remaining:
        raise RuntimeError(
            "portable family contains run-specific absolute paths: "
            + ", ".join(remaining[:10])
        )


def _evaluate_prefix_identity(
    old_family_path: Path,
    new_family_path: Path,
) -> dict[str, Any]:
    old = ActiveOnlySigned2DShieldingKernelFamily.from_json(old_family_path)
    new = ActiveOnlySigned2DShieldingKernelFamily.from_json(new_family_path)
    # Every physical 5 um topology quantum plus every midpoint is sampled.  Exact
    # identity for the continuum between samples follows constructively because
    # the new resolver delegates the whole closed prefix to an identical four-
    # state resolver, not merely because these samples happen to agree.
    quantum = np.arange(0.0, 415.0 + 2.5, 5.0)
    midpoints = quantum[:-1] + 2.5
    checks = np.sort(np.concatenate((quantum, midpoints)))
    maximum_absolute_difference = 0.0
    for extension_um in checks:
        kwargs = {
            "r_eff_over_r0": 7.25,
            "opening_strength_fraction": 0.625,
            "crack_extension_m": 1.0e-6 * float(extension_um),
        }
        old_active, old_wake = old.resolve(**kwargs)
        new_active, new_wake = new.resolve(**kwargs)
        pairs = (
            (old_active, new_active),
            (old_wake, new_wake),
            (old.active_kernel_II, new.active_kernel_II),
            (old.wake_kernel_II, new.wake_kernel_II),
            (old._last_weights, new._last_weights[: len(old._last_weights)]),
        )
        for left, right in pairs:
            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"old-domain family identity failed at {extension_um} um"
                )
            if left.size:
                maximum_absolute_difference = max(
                    maximum_absolute_difference,
                    float(np.max(np.abs(left - right))),
                )
    return {
        "schema": "pf_branching_append_only_old_domain_identity_v1",
        "passed": True,
        "domain_um": [0.0, 415.0],
        "sample_count": int(checks.size),
        "sample_policy": "all_5um_physical_quanta_and_all_2p5um_midpoints",
        "maximum_absolute_difference": maximum_absolute_difference,
        "exact_array_equality": True,
        "continuum_identity_basis": (
            "all closed-domain queries delegate to a resolver constructed from "
            "the byte-identical legacy state prefix with unchanged interpolation metadata"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    args = parser.parse_args()

    # Use the already-qualified compatibility adapter.  NumPy 2 removed the
    # legacy two-component np.cross overload; the scalar determinant is exactly
    # the intended two-dimensional segment-orientation operation.
    install_numpy2_orientation_compat()

    old_root = args.old_root.expanduser().resolve()
    outroot = args.outroot.expanduser().resolve()
    if outroot.exists():
        raise FileExistsError(f"refusing to overwrite append-only output root: {outroot}")
    old_family = old_root / "family.json"
    old_configuration = old_root / "mechanical_configuration.json"
    if sha256_file(old_family) != EXPECTED_OLD_FAMILY_SHA256:
        raise RuntimeError("historical family hash mismatch")
    if sha256_file(old_configuration) != EXPECTED_CONFIGURATION_SHA256:
        raise RuntimeError("mechanical-configuration hash mismatch")
    configuration = load_configuration(old_configuration)
    if configuration.fingerprint() != EXPECTED_CONFIGURATION_FINGERPRINT:
        raise RuntimeError("mechanical-configuration fingerprint mismatch")
    if not math.isclose(configuration.theta_deg, 40.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("append-only builder is sealed to theta=40 degrees")

    old_family_payload = _load(old_family)
    observed_old_levels = tuple(
        1.0e6 * float(row["crack_extension_m"])
        for row in old_family_payload.get("states", [])
    )
    if not np.allclose(observed_old_levels, OLD_LEVELS_UM, rtol=0.0, atol=1.0e-9):
        raise RuntimeError(f"historical family levels differ: {observed_old_levels}")

    outroot.mkdir(parents=True)
    shutil.copyfile(old_configuration, outroot / "mechanical_configuration.json")
    producer_commit = _git_commit()

    new_only_snapshots = outroot / "new_prescribed_geometry_snapshots"
    build_prescribed_geometry_snapshots(
        configuration,
        required_max_extension_um=NEW_LEVELS_UM[-1],
        outroot=new_only_snapshots,
        reference_opening_strain=1.0e-5,
        explicit_extension_levels_um=NEW_LEVELS_UM,
        mesh_seed_indices=(4, 5, 6, 7),
    )

    snapshots = outroot / "prescribed_geometry_snapshots"
    snapshots.mkdir()
    preserved_snapshot_evidence = []
    for state_id in OLD_STATE_IDS:
        preserved_snapshot_evidence.append(
            _copy_and_verify(
                old_root / "prescribed_geometry_snapshots" / state_id,
                snapshots / state_id,
            )
        )
    for state_id in NEW_STATE_IDS:
        shutil.copytree(
            new_only_snapshots / state_id,
            snapshots / state_id,
            copy_function=shutil.copyfile,
        )

    old_capture_manifest = _load(
        old_root / "prescribed_geometry_snapshots" / "kernel_capture_manifest.json"
    )
    new_capture_manifest = _load(
        new_only_snapshots / "kernel_capture_manifest.json"
    )
    state_rows = _combined_state_rows(
        snapshots, old_capture_manifest, new_capture_manifest
    )
    anchor_plan = {
        "schema": "v10.2.28_prescribed_geometry_anchor_plan_v1",
        "required_max_extension_um": 745.0,
        "atlas_anchor_spacing_m": configuration.atlas_anchor_spacing_m,
        "da_phys_m": configuration.da_phys_m,
        "explicit_extension_levels_um": list(ALL_LEVELS_UM),
        "append_only_preserved_levels_um": list(OLD_LEVELS_UM),
        "newly_measured_levels_um": list(NEW_LEVELS_UM),
        "mesh_seed_indices_by_state": {
            **dict(zip(OLD_STATE_IDS, range(4))),
            **dict(zip(NEW_STATE_IDS, range(4, 8))),
        },
        "anchors": state_rows,
    }
    _write(snapshots / "prescribed_geometry_anchor_plan.json", anchor_plan)
    capture_manifest = {
        "schema": "v10.2.28_prescribed_geometry_snapshot_manifest_v1",
        "model_id": "v10.2.28_direct_prescribed_geometry_fem_states_v1",
        "mechanical_configuration": configuration.canonical_payload(),
        "mechanical_configuration_fingerprint": configuration.fingerprint(),
        "state_count": len(state_rows),
        "states": state_rows,
        "direct_prescribed_geometry": True,
        "append_only_extension": True,
        "preserved_state_payloads_byte_identical": True,
        "prior_kernel_family_required": False,
        "material_parameter_option_required": False,
        "hazard_seed_required": False,
        "fracture_hazard_advanced": False,
        "source_emission_advanced": False,
        "moving_process_zone_advanced": False,
        "stochastic_trajectory_required": False,
        "production_physics_modified": False,
        "elasticity": new_capture_manifest["elasticity"],
    }
    _write(snapshots / "kernel_capture_manifest.json", capture_manifest)
    _write(
        snapshots / "capture_complete.json",
        {
            "schema": "v10.2.28_direct_prescribed_geometry_complete_v1",
            "complete": True,
            "state_count": len(state_rows),
        },
    )

    loads = outroot / "load_invariance"
    loads.mkdir()
    preserved_load_evidence = []
    for state_id in OLD_STATE_IDS:
        preserved_load_evidence.append(
            _copy_and_verify(old_root / "load_invariance" / state_id, loads / state_id)
        )
    for state_id in NEW_STATE_IDS:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_v10_2_14_active_load_invariance.py"),
                "--snapshot",
                str(snapshots / state_id),
                "--outroot",
                str(loads / state_id),
                "--load-scales",
                "0.5",
                "1.0",
                "1.5",
                "--magnitudes",
                "0.25",
                "0.50",
                "--linearity-tolerance",
                "0.03",
                "--load-invariance-tolerance",
                "0.05",
                "--minimum-residual-stiffness-fraction",
                "0.001",
                "--minimum-station-spacing-m",
                f"{configuration.process_zone_length_m:.17g}",
            ]
        )

    portable = outroot / "portable_assembly"
    portable.mkdir()
    family_out = outroot / "family.json"
    _run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "build_v10_2_27_kernel_from_mechanics_artifacts_v10228.py"
            ),
            "--snapshot-root",
            str(snapshots),
            "--load-invariance-root",
            str(loads),
            "--mechanical-config",
            str(old_configuration),
            "--outroot",
            str(portable),
            "--family-out",
            str(family_out),
            "--target-extension-um",
            "300",
            "--theta-deg",
            "40",
            "--da-phys-um",
            "5",
            "--event-minimum-factor",
            "0.5",
            "--event-maximum-factor",
            "4",
            "--margin-events",
            "1",
        ]
    )

    family_payload = _load(family_out)
    _validate_prefix_state_rows(old_family_payload, family_payload)
    _canonicalize_family_provenance(family_payload, loads=loads)
    family_payload.update(
        {
            "mechanical_configuration": configuration.canonical_payload(),
            "mechanical_configuration_fingerprint": configuration.fingerprint(),
            "kernel_provider_id": "v10.2.28_direct_prescribed_geometry_fem_v1",
            "kernel_recalculated_from_current_configuration": True,
            "direct_prescribed_geometry": True,
            "append_only_extension": True,
            "historical_reference_condition_required": False,
            "prior_kernel_family_required": False,
            "material_parameter_option_required": False,
            "hazard_seed_required": False,
            "stochastic_trajectory_required": False,
            "production_moving_process_zone_physics_preserved": True,
            "production_physics_modified": False,
            "capture_endpoint_reconstruction_is_measurement_only": False,
            "geometry_anchors_are_prescribed_not_captured": True,
            "crack_extension_m_semantics": "cumulative_shared_process_coordinate_m",
            "branched_shielding_geometry_newly_validated": False,
            "append_only_legacy_domain_policy": {
                "model_id": "v10.2.14_exact_legacy_domain_prefix_v1",
                "legacy_domain_max_crack_extension_m": 415.0e-6,
                "legacy_state_ids": list(OLD_STATE_IDS),
                "legacy_family_sha256": EXPECTED_OLD_FAMILY_SHA256,
                "new_states_excluded_from_legacy_domain_interpolation": True,
            },
        }
    )
    _write(family_out, family_payload)
    family_audit = validate_family(
        family_out,
        expected_configuration_fingerprint=configuration.fingerprint(),
    )
    identity = _evaluate_prefix_identity(old_family, family_out)
    _write(outroot / "old_domain_family_evaluation_identity.json", identity)

    load_rows = []
    for state_id in OLD_STATE_IDS + NEW_STATE_IDS:
        report_path = loads / state_id / "frozen_geometry_load_invariance.json"
        report = _load(report_path)
        passed = bool(
            report.get("load_invariance_passed") is True
            and report.get("checks", {}).get(
                "within_load_sign_amplitude_linearity_passed"
            )
            is True
        )
        if not passed:
            raise RuntimeError(f"load-invariance qualification failed for {state_id}")
        load_rows.append(
            {
                "state_id": state_id,
                "new_measurement": state_id in NEW_STATE_IDS,
                "passed": True,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "maximum_relative_load_variation": report["checks"][
                    "maximum_relative_load_variation"
                ],
                "maximum_within_load_relative_spread": report["checks"][
                    "maximum_within_load_relative_spread"
                ],
            }
        )

    direct_validation = {
        "schema": "pf_branching_theta40_append_only_direct_validation_v1",
        "passed": True,
        "configuration_fingerprint": configuration.fingerprint(),
        "family": str(family_out.resolve()),
        "family_sha256": family_audit["file_sha256"],
        "family_physics_fingerprint": family_audit["physics_fingerprint"],
        "preserved_state_payloads_byte_identical": True,
        "preserved_load_payloads_byte_identical": True,
        "old_domain_family_evaluation_exact": True,
        "direct_load_invariance_passed_at_every_new_state": True,
        "states": load_rows,
    }
    _write(outroot / "direct_kernel_validation_manifest.json", direct_validation)

    coverage = {
        "schema": "pf_branching_terminal_topology_signed_kernel_coverage_v5_3",
        "passed": True,
        "policy": "strict_measured_envelope_no_clipping_endpoint_hold_or_extrapolation",
        "scope": "exact_terminal_two_front_topology_only_not_universal",
        "shared_process_query_upper_bound_um": 740.0,
        "topology_quantum_guard_um": 5.0,
        "measured_endpoint_um": 745.0,
        "measured_levels_um": list(ALL_LEVELS_UM),
        "cumulative_shared_process_coordinate_model": True,
        "full_branched_shielding_geometry_validated": False,
    }
    _write(outroot / "terminal_topology_coverage_qualification.json", coverage)

    preservation = {
        "schema": "pf_branching_append_only_payload_preservation_v1",
        "passed": True,
        "old_family": str(old_family.resolve()),
        "old_family_sha256": EXPECTED_OLD_FAMILY_SHA256,
        "old_configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "snapshot_states": preserved_snapshot_evidence,
        "load_invariance_states": preserved_load_evidence,
        "family_prefix_state_rows_exact": True,
    }
    _write(outroot / "append_only_payload_preservation.json", preservation)

    final_manifest = {
        "schema": SCHEMA,
        "passed": True,
        "producer_code_commit": producer_commit,
        "old_root": str(old_root),
        "outroot": str(outroot),
        "old_family_sha256": EXPECTED_OLD_FAMILY_SHA256,
        "new_family_sha256": family_audit["file_sha256"],
        "new_family_physics_fingerprint": family_physics_fingerprint(family_out),
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "configuration_fingerprint": EXPECTED_CONFIGURATION_FINGERPRINT,
        "extension_levels_um": list(ALL_LEVELS_UM),
        "preserved_extension_levels_um": list(OLD_LEVELS_UM),
        "newly_measured_extension_levels_um": list(NEW_LEVELS_UM),
        "new_deterministic_fem_state_count": len(NEW_LEVELS_UM),
        "pf_runs_launched": 0,
        "stochastic_evolution_invoked": False,
        "old_payload_byte_identity_passed": True,
        "old_domain_family_evaluation_identity_passed": True,
        "direct_load_invariance_passed": True,
        "no_clipping_endpoint_hold_or_extrapolation": True,
        "cumulative_shared_process_coordinate_model": True,
        "full_branched_shielding_geometry_newly_validated": False,
        "family": str(family_out.resolve()),
        "family_sha256": sha256_file(family_out),
        "direct_validation": str(
            (outroot / "direct_kernel_validation_manifest.json").resolve()
        ),
        "payload_preservation": str(
            (outroot / "append_only_payload_preservation.json").resolve()
        ),
        "old_domain_identity": str(
            (outroot / "old_domain_family_evaluation_identity.json").resolve()
        ),
        "coverage_qualification": str(
            (outroot / "terminal_topology_coverage_qualification.json").resolve()
        ),
    }
    _write(outroot / "append_only_kernel_build_manifest.json", final_manifest)
    print(json.dumps(final_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
