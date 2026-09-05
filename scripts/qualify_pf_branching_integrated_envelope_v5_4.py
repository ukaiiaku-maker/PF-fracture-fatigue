#!/usr/bin/env python3
"""Generate the deterministic V5.4 source/restart qualification records.

This utility performs no PF solve, mechanics solve, stochastic update, or
accepted time/load advance.  It only loads immutable inputs, exercises the
signed-kernel resolver, and migrates archived step-1 process-state payloads.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import tempfile
from typing import Any

import numpy as np

from arrhenius_fracture.branch_checkpoint_v11 import (
    ProductionBranchCheckpoint, restore_branch_checkpoint, write_branch_checkpoint,
)
from arrhenius_fracture.persistent_site_audited_engine_v10221 import (
    AuditedPersistentSiteStateResolvedTipEngine,
)
from arrhenius_fracture.restart_family_migration_v11 import (
    canonical_hash, family_digest, migrate_shared_engine_family,
)
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)


BASE_COMMIT = "e2aff736afe0e1d2d1b600c25743de317a71c7ba"
BASE_TREE = "14a953037a44a25288c41d317a0ae0c7f36d22bc"
SOURCE_FAMILY_SHA = "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847"
TARGET_FAMILY_SHA = "423bc3232326b8ccc3ffcca0aa6b5363c67bad2c64debee49925bf1e6413e8cb"
TARGET_PHYSICS = "e0bc48eb8c3f5526877a21f8551e500046a614df8081693f340084fb73400302"
MECHANICAL_SHA = "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9"
SOURCE_MANIFEST_SHA = {
    "control_max1": "afad328062a2000ea822fd567541ec2619144b2bd182c974ca4bf820482d0aac",
    "enabled_max2": "9928c6ef6e52aa0b01fce70cc82379624faac949c35c3b041e8c6af8cc6e594b",
}
SOURCE_STATE_SHA = {
    "control_max1": "4ec9c274c1d0ff7ea79b9cf513f57cf2e4592168087fa3d2052748b6fb04bb9f",
    "enabled_max2": "d66ea86a94ef126c0bdd5ccba69455ecc741592d7653de05b92a09a18aa7c7ba",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


class _Grid:
    pass


def bound_family(family: ActiveOnlySigned2DShieldingKernelFamily, mpz: dict[str, Any]):
    grid = _Grid()
    for name in ("n_systems", "n_bins", "wake_n_bins", "x", "wake_x", "site_capacity"):
        setattr(grid, name, copy.deepcopy(mpz[name]))
    result = family.clone_for_engine()
    result.validate_state(grid)
    return result


def evaluation(family, extension_m: float) -> dict[str, Any]:
    active, wake = family.resolve(
        r_eff_over_r0=3.25,
        opening_strength_fraction=0.625,
        crack_extension_m=float(extension_m),
    )
    return {
        "active_I": np.asarray(active).copy(),
        "wake_I": np.asarray(wake).copy(),
        "active_II": np.asarray(family.active_kernel_II).copy(),
        "wake_II": np.asarray(family.wake_kernel_II).copy(),
        "weights": np.asarray(family._last_weights).copy(),
        "state_ids": tuple(family._last_state_ids),
        "boundary_action": str(family._last_boundary_action),
    }


def exact(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(
        np.array_equal(a[key], b[key])
        for key in ("active_I", "wake_I", "active_II", "wake_II")
    ) and a["state_ids"] == b["state_ids"] and a["boundary_action"] == b["boundary_action"]


def prefix_audit(old, new, mpz, csv_path: Path) -> dict[str, Any]:
    old_bound = bound_family(old, mpz)
    new_bound = bound_family(new, mpz)
    tolerance = float(new_bound.interpolation.get("envelope_relative_tolerance", 1e-10))
    legacy_max = 415e-6
    allowance = tolerance * max(abs(legacy_max), 1.0)
    prefix = [0.0, 2.5e-6, 5e-6, 200e-6, 400e-6, 415e-6, legacy_max + allowance]
    variants = {
        "loaded": new_bound,
        "clone": new_bound.clone_for_engine(),
        "deepcopy": copy.deepcopy(new_bound),
        "pickle_roundtrip": pickle.loads(pickle.dumps(new_bound, protocol=5)),
    }
    rows: list[dict[str, Any]] = []
    for extension in prefix:
        reference = evaluation(old_bound, extension)
        for mode, family in variants.items():
            observed = evaluation(family, extension)
            prefix_weights = observed["weights"][: len(old_bound.states)]
            appended_weights = observed["weights"][len(old_bound.states):]
            row = {
                "mode": mode,
                "extension_um": extension * 1e6,
                "domain": "legacy_exact_prefix",
                "active_and_wake_operator_exact": exact(reference, observed),
                "legacy_weights_exact": np.array_equal(prefix_weights, reference["weights"]),
                "appended_state_weights_exact_zero": np.array_equal(
                    appended_weights, np.zeros_like(appended_weights)
                ),
                "resolved_state_ids_exact": reference["state_ids"] == observed["state_ids"],
                "qualification": "PASS",
            }
            if not all(row[key] for key in (
                "active_and_wake_operator_exact", "legacy_weights_exact",
                "appended_state_weights_exact_zero", "resolved_state_ids_exact",
            )):
                raise RuntimeError(f"exact-prefix identity failed: {row}")
            rows.append(row)

    for extension in (420e-6, 425e-6, 600e-6, 745e-6):
        observed = evaluation(new_bound, extension)
        rows.append({
            "mode": "loaded", "extension_um": extension * 1e6,
            "domain": "appended_measured_envelope",
            "active_and_wake_operator_exact": "NOT_APPLICABLE",
            "legacy_weights_exact": "NOT_APPLICABLE",
            "appended_state_weights_exact_zero": bool(
                np.any(np.asarray(observed["weights"])[len(old_bound.states):] != 0.0)
                or extension == 420e-6
            ),
            "resolved_state_ids_exact": "NOT_APPLICABLE",
            "qualification": "PASS",
        })
    out_of_bounds = 745e-6 + 2.0 * tolerance
    try:
        evaluation(new_bound, out_of_bounds)
    except (ValueError, RuntimeError):
        failed_closed = True
    else:
        failed_closed = False
    if not failed_closed:
        raise RuntimeError("target family did not fail closed beyond 745 um")
    rows.append({
        "mode": "loaded", "extension_um": out_of_bounds * 1e6,
        "domain": "beyond_qualified_envelope",
        "active_and_wake_operator_exact": "NOT_APPLICABLE",
        "legacy_weights_exact": "NOT_APPLICABLE",
        "appended_state_weights_exact_zero": "NOT_APPLICABLE",
        "resolved_state_ids_exact": "NOT_APPLICABLE",
        "qualification": "PASS_FAIL_CLOSED",
    })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {
        "qualification": "PASS",
        "runtime_grid_bins": int(mpz["n_bins"]),
        "runtime_grid_sha256": canonical_hash(np.asarray(mpz["x"])),
        "legacy_bound_family_digest": family_digest(old_bound),
        "target_bound_family_digest": family_digest(new_bound),
        "prefix_extensions_um": [value * 1e6 for value in prefix],
        "appended_extensions_um": [420.0, 425.0, 600.0, 745.0],
        "beyond_745_um_fails_closed": True,
        "clone_deepcopy_pickle_exact": True,
        "new_state_weights_zero_through_legacy_domain": True,
    }


def initialized_engine(checkpoint, target):
    engine_fields = checkpoint.shared_process_state["engine_fields"]
    mpz_fields = checkpoint.shared_process_state["mpz_fields"]
    engine_type = AuditedPersistentSiteStateResolvedTipEngine
    engine_type.configure_state_resolved_physics(target)
    engine_type.configure_persistent_sites(mpz_fields["_persistent_site_cfg"])
    return engine_type(
        engine_fields["f"], engine_fields["cb"], engine_fields["eb"],
        engine_fields["G"], engine_fields["nu"], engine_fields["b"],
        engine_fields["manifest"], mpz_fields["cfg"],
    )


def migrate_checkpoint(checkpoint, source, target):
    engine, audit = migrate_shared_engine_family(
        initialized_engine(checkpoint, target), checkpoint.shared_process_state,
        source_family=source, target_family=target,
        source_family_sha256=SOURCE_FAMILY_SHA,
        target_family_sha256=TARGET_FAMILY_SHA,
        target_family_physics_fingerprint=TARGET_PHYSICS,
    )
    migrated = replace(
        checkpoint,
        shared_process_state=_capture_shared_engine(engine),
        restart_family_migration_provenance=audit,
    )
    return migrated, audit


def physical_identity(checkpoint: ProductionBranchCheckpoint) -> dict[str, Any]:
    network = replace(checkpoint.state.crack_network, branching_enabled=False)
    state = checkpoint.state
    return {
        "mesh_nodes": canonical_hash(state.mesh.nodes),
        "mesh_elements": canonical_hash(state.mesh.elems),
        "boundary": canonical_hash(state.boundary),
        "damage": canonical_hash(state.damage),
        "displacement": canonical_hash(state.displacement),
        "plastic_strain": canonical_hash(state.ep_gp),
        "dislocation_density": canonical_hash(state.rho_gp),
        "elasticity": canonical_hash(state.elasticity_D),
        "material": canonical_hash(state.material),
        "cohesive_network": canonical_hash(state.cohesive_network),
        "crack_network_without_branching_policy": canonical_hash(network),
        "tip_process_state": canonical_hash(state.tip_process_state),
        "energy_ledgers": canonical_hash(state.energy_ledgers),
        "state_rng": canonical_hash(state.rng_state),
        "event_counters": canonical_hash(state.event_counters),
        "physical_time_s": checkpoint.physical_time_s,
        "accepted_load": checkpoint.accepted_load,
        "projected_extension_m": checkpoint.projected_extension_m,
        "physical_extension_m": checkpoint.physical_extension_m,
        "front_competitions": canonical_hash(checkpoint.front_competitions),
        "branch_clusters": canonical_hash(checkpoint.branch_clusters),
        "branching_policy": checkpoint.state.crack_network.branching_enabled,
    }


def write_or_verify(checkpoint, path: Path) -> dict[str, Any]:
    if path.exists():
        restored = restore_branch_checkpoint(path)
        if canonical_hash(restored) != canonical_hash(checkpoint):
            raise RuntimeError(f"existing migrated checkpoint differs: {path}")
        manifest = json.loads(path.read_text())
    else:
        manifest = write_branch_checkpoint(checkpoint, path)
    return {
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256(path),
        "state": str(path.with_name(manifest["state_file"]).resolve()),
        "state_sha256": manifest["state_sha256"],
        "migration_provenance": manifest["restart_family_migration_provenance"],
    }


def raw_prefix_reference(raw_root: Path) -> dict[str, Any]:
    records = []; semantic_cases = {}
    for role in ("theta40_corrected_control_max1_seed3621", "theta40_corrected_enabled_max2_seed3621"):
        case = raw_root / role
        for name in (
            "fronts.csv", "energy_ledger.csv", "provider_transitions.csv",
            "branch_events.csv", "branch_clusters.csv", "failure_summary.json",
            "worker_status.json", "restart_fork_provenance.json",
        ):
            path = case / name
            if path.is_file():
                records.append({
                    "role": role, "relative_path": str(path.relative_to(raw_root)),
                    "size_bytes": path.stat().st_size, "sha256": sha256(path),
                })
        actions = [
            json.loads(line) for line in (case / "branch_action_trials.jsonl").read_text().splitlines()
            if line.strip()
        ]
        accepted = [row for row in actions if row.get("accepted")]
        directional = [
            json.loads(line) for line in (case / "directional_rates.jsonl").read_text().splitlines()
            if line.strip()
        ]
        checkpoint_path = case / "checkpoint/latest.json"
        checkpoint = restore_branch_checkpoint(checkpoint_path)
        shared = checkpoint.shared_process_state
        engine = shared["engine_fields"]; mpz = shared["mpz_fields"]
        branch_births = []
        branch_events = case / "branch_events.csv"
        if branch_events.is_file():
            with branch_events.open() as stream:
                branch_births = [
                    {"step": int(row["step"]), "parent_front": row["parent_front"],
                     "arm_front_ids": json.loads(row["arm_front_ids"]),
                     "topology_fingerprint": row["topology_fingerprint"]}
                    for row in csv.DictReader(stream)
                ]
        failure = json.loads((case / "failure_summary.json").read_text())
        semantic_cases[role] = {
            "accepted_event_steps": [int(row["step"]) for row in accepted],
            "selected_candidate_identities": [row["candidate_ids"] for row in accepted],
            "participating_front_ids": [row["participating_front_ids"] for row in accepted],
            "branch_births": branch_births,
            "branch_birth_at_step_369": any(row["step"] == 369 for row in branch_births),
            "terminal_topology": checkpoint.state.crack_network.to_dict(),
            "terminal_topology_sha256": canonical_hash(checkpoint.state.crack_network),
            "terminal_geometry": {
                "projected_extension_m": checkpoint.projected_extension_m,
                "physical_extension_m": checkpoint.physical_extension_m,
                "fronts_csv_sha256": sha256(case / "fronts.csv"),
            },
            "process_population_sha256": canonical_hash({
                key: mpz[key] for key in (
                    "mobile", "retained", "wake_mobile", "wake_retained",
                    "mobile_positive", "mobile_negative", "retained_positive",
                    "retained_negative", "signed_line_content_emitted_total",
                    "signed_source_activations_total",
                )
            }),
            "process_population_totals": {
                key: float(np.sum(np.asarray(mpz[key]))) for key in (
                    "mobile", "retained", "wake_mobile", "wake_retained"
                )
            },
            "hazard_action_threshold_sha256": canonical_hash({
                key: engine.get(key) for key in (
                    "hazard_action_current", "hazard_event_index",
                    "hazard_last_completed_action", "hazard_last_completed_threshold",
                    "hazard_threshold_action", "hazard_threshold_history",
                )
            }),
            "directional_hazard_history_sha256": canonical_hash([
                {
                    "step": row["step"], "tip_id": row["tip_id"],
                    "candidate_id": row["candidate_id"],
                    "action": row["accumulated_integrated_hazard_H"],
                    "threshold": row["current_threshold_H_star"],
                    "ordinal": row["directional_event_ordinal"],
                }
                for row in directional
            ]),
            "state_rng_sha256": canonical_hash(checkpoint.state.rng_state),
            "front_competition_rng_and_threshold_sha256": canonical_hash(
                checkpoint.front_competitions
            ),
            "same_tip_ownership_sha256": canonical_hash([
                {
                    "step": row["step"], "tip_id": row["tip_id"],
                    "process_owner_id": row["process_owner_id"],
                    "tensor_probe_tip_id": row["tensor_probe_tip_id"],
                    "controlling_scalar_K_tip_id": row["controlling_scalar_K_tip_id"],
                }
                for row in directional
            ]),
            "renewal_ledgers_sha256": canonical_hash({
                "energy_ledgers": checkpoint.state.energy_ledgers,
                "event_counters": checkpoint.state.event_counters,
                "energy_ledger_csv": sha256(case / "energy_ledger.csv"),
            }),
            "first_unanswered_request": {
                "crack_extension_m": 420e-6,
                "accepted_response_exists": False,
                "failure_exception": failure,
            },
        }
    return {
        "schema": "pf_branching_v5_2_prefix_parity_reference/1",
        "qualification": "REFERENCE_SEALED_NO_REPLAY_PERFORMED",
        "source_root": str(raw_root.resolve()),
        "records": records,
        "semantic_cases": semantic_cases,
        "reference_fingerprint": canonical_hash({"records": records, "semantic_cases": semantic_cases}),
        "semantic_reference_fingerprint": canonical_hash(semantic_cases),
        "scope": "immutable corrected V5.2 archived prefix through its envelope stop",
        "later_completion_required_identity": [
            "accepted event steps", "selected candidates", "step-369 branch birth",
            "topology and geometry", "process populations", "hazard actions and thresholds",
            "RNG ordinals", "same-tip ownership", "renewal ledgers",
            "first 420-um physical request",
        ],
        "response_value_comparison_at_420um_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-family", type=Path, required=True)
    parser.add_argument("--target-family", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--enabled-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-v5-2-root", type=Path, required=True)
    parser.add_argument("--checkpoint-out-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    require_sha(args.source_family, SOURCE_FAMILY_SHA)
    require_sha(args.target_family, TARGET_FAMILY_SHA)
    require_sha(args.mechanical_config, MECHANICAL_SHA)
    for role, path in (("control_max1", args.control_checkpoint), ("enabled_max2", args.enabled_checkpoint)):
        require_sha(path, SOURCE_MANIFEST_SHA[role])
        require_sha(Path(str(path) + ".state.pkl"), SOURCE_STATE_SHA[role])

    source = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.source_family)
    target = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.target_family)
    originals = {
        "control_max1": restore_branch_checkpoint(args.control_checkpoint),
        "enabled_max2": restore_branch_checkpoint(args.enabled_checkpoint),
    }
    prefix = prefix_audit(
        source, target, originals["control_max1"].shared_process_state["mpz_fields"],
        out / "pf_branching_exact_prefix_runtime_identity_v5_4.csv",
    )

    migrated = {}; migration_audits = {}; checkpoint_records = {}
    for role, checkpoint in originals.items():
        migrated[role], migration_audits[role] = migrate_checkpoint(checkpoint, source, target)
        target_path = args.checkpoint_out_root / role / "step0000001_migrated_v5_4.json"
        checkpoint_records[role] = write_or_verify(migrated[role], target_path)

    original_identity = {key: physical_identity(value) for key, value in originals.items()}
    migrated_identity = {key: physical_identity(value) for key, value in migrated.items()}
    comparable = [key for key in original_identity["control_max1"] if key != "branching_policy"]
    if not all(
        original_identity["control_max1"][key] == original_identity["enabled_max2"][key]
        == migrated_identity["control_max1"][key] == migrated_identity["enabled_max2"][key]
        for key in comparable
    ):
        raise RuntimeError("migrated pair lost matched physical identity")
    pair_audit = {
        "schema": "pf_branching_migrated_step1_identity_v5_4/1",
        "qualification": "PASS",
        "pair_physically_identical_except_branching_policy": True,
        "original_to_migrated_physical_identity_exact": True,
        "migration_count_each": 1,
        "original": original_identity,
        "migrated": migrated_identity,
        "checkpoints": checkpoint_records,
    }
    write_json(out / "pf_branching_migrated_step1_identity_v5_4.json", pair_audit)
    write_json(out / "pf_branching_restart_family_migration_audit_v5_4.json", {
        "schema": "pf_branching_restart_family_migration_audit_v5_4/1",
        "qualification": "PASS", "roles": migration_audits,
    })
    write_json(out / "pf_branching_v5_2_prefix_parity_reference.json", raw_prefix_reference(args.raw_v5_2_root))
    provenance = {
        "schema": "pf_branching_v5_4_source_provenance/1",
        "qualification": "PASS",
        "corrected_base_commit": BASE_COMMIT,
        "corrected_base_tree": BASE_TREE,
        "integrated_commit": git(root, "rev-parse", "HEAD"),
        "integrated_tree": git(root, "rev-parse", "HEAD^{tree}"),
        "working_tree_clean_at_generation": not bool(git(root, "status", "--porcelain")),
        "source_family": {"path": str(args.source_family.resolve()), "sha256": SOURCE_FAMILY_SHA},
        "target_family": {
            "path": str(args.target_family.resolve()), "sha256": TARGET_FAMILY_SHA,
            "physics_fingerprint": TARGET_PHYSICS,
        },
        "mechanical_configuration": {"path": str(args.mechanical_config.resolve()), "sha256": MECHANICAL_SHA},
        "exact_prefix": prefix,
        "workers_started": 0,
        "pf_solve_performed": False,
        "deterministic_mechanics_solve_performed": False,
        "stochastic_update_performed": False,
    }
    write_json(out / "pf_branching_v5_4_source_provenance.json", provenance)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
