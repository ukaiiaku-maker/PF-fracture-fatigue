#!/usr/bin/env python3
"""Qualify recursive owner transitions without manufacturing a branch event.

The portable fixture remains an ownership regression.  When an accepted V12
checkpoint is supplied, the same report also records the native birth,
daughter evolution, physical owner-connectivity gate, and exact continuation
command.  It never promotes an injected transition to a physical event.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.qualification_manifest import capacity_preflight
from arrhenius_fracture.process_state_ownership_v11 import (
    ProcessStateOwner, ProcessStateOwnerRegistry,
)
from arrhenius_fracture.multifront_checkpoint_v12 import (
    load_accepted_boundary_checkpoint_v12,
)


def qualification_fixture() -> dict:
    registry = ProcessStateOwnerRegistry(
        {"root": "root"},
        {"root": ProcessStateOwner(
            "root", "single_tip_engine", process_engine_id="root"
        )},
    )
    first = registry.branch("root", "cluster-1", ("daughter-1", "daughter-2"))
    first.validate(
        ("daughter-1", "daughter-2"), engine_ids=("cluster-1",),
        cluster_ids=("cluster-1",), reservoir_ids=(),
    )
    resolved = first.resolve(
        "cluster-1", ("daughter-1", "daughter-2"), "reservoir-1"
    )
    resolved.validate(
        ("daughter-1", "daughter-2"),
        engine_ids=("daughter-1", "daughter-2"), cluster_ids=(),
        reservoir_ids=("reservoir-1",),
    )
    second = resolved.branch(
        "daughter-1", "cluster-2", ("granddaughter-1", "granddaughter-2")
    )
    active = ("daughter-2", "granddaughter-1", "granddaughter-2")
    second.validate(
        active, engine_ids=("daughter-2", "cluster-2"),
        cluster_ids=("cluster-2",), reservoir_ids=("reservoir-1",),
    )
    serialized = second.to_dict()
    restored = ProcessStateOwnerRegistry.from_dict(
        json.loads(json.dumps(serialized, sort_keys=True))
    )
    restored.validate(
        active, engine_ids=("daughter-2", "cluster-2"),
        cluster_ids=("cluster-2",), reservoir_ids=("reservoir-1",),
    )
    if restored.to_dict() != serialized:
        raise RuntimeError("recursive owner registry changed across checkpoint encoding")
    return {
        "schema": "recursive-branch-owner-adapter-qualification/1",
        "fixture_classification": "INJECTED_OWNER_TRANSITIONS_NOT_NATURAL_BRANCH_PREDICTION",
        "transitions": [
            "single_owner_to_unresolved_first_generation_cluster",
            "first_generation_cluster_to_independent_daughter_owners",
            "resolved_daughter_to_unresolved_second_generation_cluster",
            "checkpoint_round_trip",
        ],
        "active_tips": list(active),
        "owner_registry": serialized,
        "checkpoint_exact": True,
        "mutable_independent_owner_aliasing": False,
        "native_birth_demonstrated": False,
        "status": "PARTIAL_ADAPTER_QUALIFIED_NATIVE_BIRTH_PENDING",
    }


def native_checkpoint_evidence(checkpoint: Path, interval: Path | None = None) -> dict:
    """Summarize an exact accepted V12 boundary without advancing its clocks."""
    restored = load_accepted_boundary_checkpoint_v12(checkpoint)
    runtime = restored.runtime
    branches = {item.branch_id: item for item in runtime.crack_network.branches}
    births = [
        item for item in runtime.transaction_records
        if item.action_type == "two_arm" and len(item.created_front_ids) == 2
    ]
    advances = [
        item for item in runtime.transaction_records
        if item.action_type == "one_arm" and item.realized_lengths_m
    ]
    active = tuple(runtime.active_front_ids)
    recursive_eligible = []
    unresolved = []
    for front_id in active:
        owner_id = runtime.owner_by_front[front_id]
        region = runtime.process_regions[owner_id]
        if len(region.member_front_ids) == 1 and not region.unresolved_junction_ids:
            recursive_eligible.append(front_id)
        else:
            unresolved.append({
                "front_id": front_id,
                "owner_id": owner_id,
                "member_front_ids": sorted(region.member_front_ids),
                "unresolved_junction_ids": sorted(region.unresolved_junction_ids),
                "reason": "parent_process_zone_still_unresolved",
            })
    observations = []
    if interval is not None:
        path = interval / "observation.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"accepted interval observations absent: {path}")
        observations = [json.loads(line) for line in path.read_text().splitlines() if line]
    imported_native_birth = bool(
        runtime.cumulative_branch_births >= 1
        and runtime.junctions
        and any(
            junction.birth_transaction_id.startswith("v11:event:")
            and all(branches[child].generation >= 1 for child in junction.child_branch_ids)
            for junction in runtime.junctions.values()
        )
        and runtime.compatibility_provenance.get("source_schema")
        == "v11.production-branch-network-checkpoint/2"
    )
    native_birth = bool(births) or imported_native_birth
    daughter_evolution = bool(
        native_birth
        and all(branches[item].generation >= 1 for item in active)
        and advances
        and any(float(item.realized_lengths_m[0]) > 0.0 for item in advances)
    )
    positive_active_rates = {
        row["front_id"]: float(row["directional_rate_per_s"])
        for row in observations
        if row.get("front_id") in active and float(row.get("directional_rate_per_s", 0.0)) > 0.0
    }
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    next_steps = max(restored.step_count + 2000, 10000)
    command = [
        sys.executable, "-m",
        "arrhenius_fracture.sharp_front_current_source_multifront_physical_v12",
        "--maximum-fronts", "4", "--v12-restart-checkpoint", str(checkpoint.resolve()),
        "--v12-stop-after-total-births", "2", "--v12-post-birth-events", "3",
        "--signed-kernel-family",
        "/Volumes/Data/Data/Nanopillar_calculation/"
        "PF-fracture-fatigue_theta40_signed_kernel_append_only_"
        "v5_3_20260901T120000Z_repro/family.json",
        "--mode", "2d", "--parameter-option",
        "v913_paper_weakT01_0129902_persistent_sites", "--temperatures", "700",
        "--steps", str(next_steps), "--nx", "36", "--ny", "72", "--dU", "2e-7",
        "--dt", "1e-6", "--n-stagger", "2", "--tip-h-fine", "1e-6",
        "--tip-ratio", "1.2", "--da-phys", "5e-6", "--target-crack-extension-um", "1000",
        "--mpz-length-um", "50", "--mpz-n-bins", "80", "--front-state-model", "moving_pz",
        "--tip-source-model", "continuum", "--tip-kinetics-mode", "moving_velocity",
        "--bulk-plasticity-mode", "tip_only", "--directional-j-mode", "root_signed",
        "--tip-plasticity", "--active-shielding", "--signed-active-shielding",
        "--mobile-shield-fraction", "0", "--no-wake-shielding", "--crystal-aniso",
        "--crystal-compete", "--crystal-theta-deg", "40", "--crystal-material", "w",
        "--j-decomposition", "cluster", "--crack-backend", "sharp_wake",
        "--adaptive-events", "--adaptive-event-target", "0.15", "--print-every", "1",
        "--out", "{new_output_directory}",
    ]
    return {
        "schema": "recursive-native-branch-readiness/2",
        "checkpoint": {
            "path": str(checkpoint.resolve()), "sha256": checkpoint_sha,
            "payload_sha256": restored.payload_sha256, "step": restored.step_count,
            "exact_current_loader_validation": True,
        },
        "native_first_birth_demonstrated": native_birth,
        "native_birth_evidence_origin": (
            "accepted_v12_transaction" if births
            else "immutable_v11_native_birth_imported_with_exact_v12_prefix"
            if imported_native_birth else None
        ),
        "mechanically_evolved_daughters_demonstrated": daughter_evolution,
        "cumulative_branch_births": runtime.cumulative_branch_births,
        "active_front_ids": list(active),
        "active_fronts": [{
            "front_id": front_id, "generation": branches[front_id].generation,
            "path_length_um": branches[front_id].physical_path_length_m * 1.0e6,
            "tip_xy_um": [value * 1.0e6 for value in branches[front_id].tip],
        } for front_id in active],
        "accepted_one_arm_advances": len(advances),
        "positive_rate_active_fronts": positive_active_rates,
        "recursive_eligible_front_ids": recursive_eligible,
        "unresolved_owner_gates": unresolved,
        "second_generation_birth_demonstrated": runtime.cumulative_branch_births >= 2,
        "next_native_command": command,
        "production_execution_authorized": False,
        "status": (
            "COMPLETE_NATIVE_RECURSIVE_BIRTH_AND_DAUGHTER_RESTART"
            if runtime.cumulative_branch_births >= 2
            else "NATIVE_FIRST_BIRTH_DAUGHTER_RESTART_QUALIFIED_"
                 "SECOND_GENERATION_PHYSICAL_OWNER_GATE_PENDING"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--second-generation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--native-checkpoint", type=Path)
    p.add_argument("--accepted-interval", type=Path)
    p.add_argument("--output")
    args = p.parse_args()
    if not args.second_generation:
        raise SystemExit("recursive readiness requires a second-generation birth")
    payload = {"qualification": "Q-BRANCH-RECURSIVE",
               "native_birth_entry": "scripts/run_v11_branching_case.py",
               "capacity": capacity_preflight(ROOT, 24.0).__dict__,
               "production_execution_authorized": False,
               **qualification_fixture()}
    if args.native_checkpoint is not None:
        payload["native_evidence"] = native_checkpoint_evidence(
            args.native_checkpoint, args.accepted_interval,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not args.dry_run and not args.output:
        raise SystemExit(
            "native branch execution remains disabled; use --output to retain "
            "the ownership-adapter qualification"
        )


if __name__ == "__main__":
    main()
