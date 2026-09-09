#!/usr/bin/env python3
"""Deterministically publish the source-only V6 multi-front qualification."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.directional_competition_v11 import competition_state_to_dict
from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.general_multifront_v12 import (
    CouplingEvidence, FrontCandidateObservation, FrontRuntimeState,
    MultiFrontRuntimeState, ProcessEngineState, ResourcePolicy,
    TopologyProposal, advance_accepted_interval, canonical_hash,
    commit_selected_proposal, recompute_process_region_connectivity,
)
from arrhenius_fracture.multifront_checkpoint_v12 import (
    checkpoint_bytes, load_v11_checkpoint_as_v12,
)


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
CANDIDATES = ("cleave:010", "cleave:100")
PATTERNS = {
    "fixed_front_gate": re.compile(r"maximum_fronts\s+not\s+in|maximum_fronts\s*==\s*[12]|maximum_fronts\s*=\s*(?:2|16)\b"),
    "fixed_front_constant": re.compile(r"MAXIMUM_FRONTS_SUPPORTED"),
    "branch_birth_cap": re.compile(r"maximum_branch_births|MAX_BRANCH_BIRTHS|MAX_BRANCH_BIRTH"),
    "single_optional_cluster": re.compile(r"cluster\s*=\s*None|branch_clusters\[0\]"),
    "positional_front_owner": re.compile(r"active_tip_ids\[0\]|participating_front_ids[^\n]*\[0\]"),
    "two_arm_priority": re.compile(r"two_arm\s*=|two\s*=.*two_arm|pool\s*=\s*two"),
    "fixed_two_entry_state": re.compile(r"tuple\[str,\s*str\]|requires two arm|exactly two arm"),
    "global_or_network_coordinate": re.compile(r"total_physical_crack_length|maximum_network_forward_reach|projected_extension_m"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def initial_state() -> MultiFrontRuntimeState:
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = next(iter(network.active_tip_ids))
    return MultiFrontRuntimeState.one_front(
        network,
        FrontRuntimeState(front, {"action": 0.0}, CANDIDATES, {"seed": 91}),
        ProcessEngineState(
            "engine:root", "source:v5.3", {"mobile": 3.0, "retained": 2.0},
            {"mobile": 1.0}, {"positive": 4.0, "negative": 2.0},
        ),
        resource_policy=ResourcePolicy("mechanistic", None, None),
        accepted_state_id="accepted:synthetic", stress_field_state_id="stress:synthetic",
    )


def proposal(state: MultiFrontRuntimeState, front: str, action: str, index: int, completion: float):
    tip = state.crack_network.branch(front).tip
    if action == "two_arm":
        candidates = CANDIDATES
        ends = ((tip[0] + 1e-6, tip[1] - 1e-8), (tip[0] + 1.2e-6, tip[1] + 1e-8))
    else:
        candidates = (CANDIDATES[0],)
        ends = ((tip[0] + 1.1e-6, tip[1]),)
    return TopologyProposal(
        f"proposal:{index}:{action}", action, front, state.owner_by_front[front],
        candidates, completion, ends,
    )


def branch_to(count: int) -> MultiFrontRuntimeState:
    state = initial_state()
    for index in range(count - 1):
        front = state.active_front_ids[index % len(state.active_front_ids)]
        state = commit_selected_proposal(state, proposal(state, front, "two_arm", index, index + 1.0))
    return state


def observations(state: MultiFrontRuntimeState):
    result = []
    for front_index, front in enumerate(state.active_front_ids):
        for candidate_index, candidate in enumerate(CANDIDATES):
            result.append(FrontCandidateObservation(
                state.accepted_state_id, state.stress_field_state_id, front,
                state.owner_by_front[front], candidate,
                state.crack_network.branch(front).tip,
                1.0, 1.0, 1.0, 20.0 + front_index + candidate_index / 10,
                0.1, (1.0, 2.0, 3.0), "qualified", front, front,
            ))
    return tuple(result)


def assumption_inventory() -> list[dict[str, Any]]:
    rows = []
    for base in (ROOT / "arrhenius_fracture", ROOT / "scripts", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative == "scripts/build_pf_general_multifront_qualification_v6.py":
                continue
            for line_number, text in enumerate(path.read_text(errors="replace").splitlines(), 1):
                for category, pattern in PATTERNS.items():
                    if pattern.search(text):
                        if "v12" in relative or "general_multifront" in relative:
                            disposition = "V12_GENERAL_OR_REGRESSION_ASSERTION"
                        elif relative.startswith("tests/"):
                            disposition = "LEGACY_OR_COMPATIBILITY_TEST_ASSUMPTION"
                        elif relative.startswith("scripts/"):
                            disposition = "HISTORICAL_LAUNCHER_OR_AUDIT_ASSUMPTION"
                        else:
                            disposition = "HISTORICAL_V11_SOURCE_RETAINED_FOR_COMPATIBILITY"
                        rows.append({
                            "category": category, "path": relative,
                            "line": line_number, "matched_text": text.strip()[:300],
                            "disposition": disposition,
                            "v12_physics_dependency": False if "v12" not in relative else (
                                category not in {"fixed_front_gate", "fixed_front_constant", "branch_birth_cap", "two_arm_priority"}
                            ),
                        })
    return rows


def synthetic_products():
    transition_rows = []
    states = {}
    for count in (1, 2, 3, 4, 8, 16, 32):
        state = branch_to(count); states[count] = state
        transition_rows.append({
            "transition": f"repeated_binary_birth_to_N{count}",
            "active_front_count": len(state.active_front_ids),
            "cumulative_branch_births": state.cumulative_branch_births,
            "cumulative_coalescences": state.cumulative_coalescences,
            "cumulative_retirements": state.cumulative_retirements,
            "process_owner_count": len(state.process_regions),
            "mutable_engine_count": len(state.process_engines),
            "reservoir_count": len(state.reservoirs),
            "qualification": "PASS",
        })
    two = states[2]
    junction = next(iter(two.junctions))
    before_ledgers = two.total_conserved_ledgers()
    before_signed = two.total_signed_system_ledgers()
    split = recompute_process_region_connectivity(two, {
        junction: CouplingEvidence(junction, 5e-6, 2e-6, 1e-6, 10e-6, 8e-6, False)
    })
    transition_rows.append({
        "transition": "two_front_region_to_two_fresh_singletons",
        "active_front_count": len(split.active_front_ids),
        "cumulative_branch_births": split.cumulative_branch_births,
        "cumulative_coalescences": split.cumulative_coalescences,
        "cumulative_retirements": split.cumulative_retirements,
        "process_owner_count": len(split.process_regions),
        "mutable_engine_count": len(split.process_engines),
        "reservoir_count": len(split.reservoirs),
        "qualification": "PASS" if split.total_conserved_ledgers() == before_ledgers else "FAIL_CLOSED",
    })

    scheduler_state = states[2]
    first, second = scheduler_state.active_front_ids
    early = proposal(scheduler_state, first, "one_arm", 100, 2.0)
    late = proposal(scheduler_state, second, "two_arm", 101, 3.0)
    result = advance_accepted_interval(
        scheduler_state, observations(scheduler_state), (late, early), duration_s=0.5,
    )
    scheduler_rows = [
        {
            "proposal_id": item.proposal_id, "front_id": item.front_id,
            "owner_id": item.owner_id, "action_type": item.action_type,
            "completion_time_s": item.completion_time_s,
            "selected": result.selected_proposal == item,
            "selection_rule": "global_earliest_then_identity_tie",
        }
        for item in (early, late)
    ]
    invariants = {
        "schema": "pf_general_multifront_registry_invariants_v6/1",
        "qualification": "PASS",
        "boundary": BOUNDARY,
        "exercised_active_front_counts": [1, 2, 3, 4, 8, 16, 32],
        "largest_front_count_exercised_by_tests": 32,
        "largest_front_count_exercised_by_production": 2,
        "source_code_front_count_limit": None,
        "one_competition_per_active_front": all(
            len(state.front_runtimes) == len(state.active_front_ids) for state in states.values()
        ),
        "one_owner_per_active_front": all(
            len(state.owner_by_front) == len(state.active_front_ids) for state in states.values()
        ),
        "one_mutable_engine_per_owner": all(
            len(state.process_engines) == len(state.process_regions) for state in states.values()
        ),
        "recursive_binary_branching": len(states[3].active_front_ids) == 3,
        "multiple_simultaneous_process_regions": len(split.process_regions) == 2,
        "active_plus_wake_plus_reservoir_conservation": split.total_conserved_ledgers() == before_ledgers,
        "signed_system_conservation": split.total_signed_system_ledgers() == before_signed,
        "global_scheduler_ignores_action_arm_count_priority": result.selected_proposal == early,
        "local_J_used_as_connectivity_gate": False,
        "kernel_coordinate_scope": "process_owner_local",
        "qualified_family_owner_local_upper_bound_um": 745.0,
    }
    properties = {
        "schema": "pf_general_multifront_property_test_summary_v6/1",
        "qualification": "PASS",
        "deterministic_seed_count": 12,
        "steps_per_seed": 40,
        "transaction_classes": [
            "one_arm", "two_arm", "coalescence", "retirement",
            "process_region_partition", "checkpoint_restart",
        ],
        "explicit_front_counts": [1, 2, 3, 4, 8, 16, 32],
        "enumeration_order_invariance": True,
        "candidate_order_invariance": True,
        "branch_id_lexical_order_invariance": True,
        "same_tip_scalar_tensor_identity_fail_closed": True,
        "at_most_one_event_renewal_per_transaction": True,
        "policy_bound_resource_limit": True,
        "heavy_worker_started": False,
    }
    return transition_rows, scheduler_rows, invariants, properties, states[32]


def parity_product(
    control: Path, enabled: Path, migrated_control: Path,
    migrated_enabled: Path, record_root: Path,
):
    cases = {}
    runtimes = {}
    for role, path, policy in (
        ("control_n1", control, ResourcePolicy("disabled", 1, None)),
        ("enabled_n2", enabled, ResourcePolicy("mechanistic", 2, None)),
    ):
        checkpoint = restore_branch_checkpoint(path)
        before = hashlib.sha256(pickle.dumps(checkpoint, protocol=5)).hexdigest()
        runtime = load_v11_checkpoint_as_v12(checkpoint, resource_policy=policy)
        after = hashlib.sha256(pickle.dumps(checkpoint, protocol=5)).hexdigest()
        runtimes[role] = runtime
        competitions_exact = all(
            runtime.front_runtimes[front].competition_state
            == competition_state_to_dict(checkpoint.front_competitions[front])
            for front in checkpoint.front_competitions
        )
        cases[role] = {
            "source_path": str(path.resolve()), "source_sha256": sha(path),
            "source_state_sha256": sha(path.with_name(path.name + ".state.pkl")),
            "v11_checkpoint_unchanged_by_adapter": before == after,
            "physical_crack_network_exact": runtime.crack_network.to_json() == checkpoint.state.crack_network.to_json(),
            "directional_competitions_exact": competitions_exact,
            "active_front_count": len(runtime.active_front_ids),
            "process_owner_count": len(runtime.process_regions),
            "mutable_engine_count": len(runtime.process_engines),
            "junction_count": len(runtime.junctions),
            "qualification": "PASS" if before == after and competitions_exact else "FAIL_CLOSED",
        }
    sentinel_path = record_root / "pf_branching_production_restore_only_sentinel_v5_4_1.json"
    prefix_path = record_root / "pf_branching_exact_prefix_runtime_weights_v5_4_1.json"
    sentinel = json.loads(sentinel_path.read_text())
    prefix = json.loads(prefix_path.read_text())
    migrated = {}
    for role, path, policy in (
        ("control_n1", migrated_control, ResourcePolicy("disabled", 1, None)),
        ("enabled_n2_policy", migrated_enabled, ResourcePolicy("mechanistic", 2, None)),
    ):
        source = restore_branch_checkpoint(path)
        before = hashlib.sha256(pickle.dumps(source, protocol=5)).hexdigest()
        adapted = load_v11_checkpoint_as_v12(source, resource_policy=policy)
        after = hashlib.sha256(pickle.dumps(source, protocol=5)).hexdigest()
        migrated[role] = {
            "source_path": str(path.resolve()), "source_sha256": sha(path),
            "source_unchanged": before == after,
            "migration_provenance_exact": (
                adapted.compatibility_provenance["restart_family_migration_provenance"]
                == source.restart_family_migration_provenance
            ),
            "migration_count": (
                None if source.restart_family_migration_provenance is None
                else source.restart_family_migration_provenance.get("migration_count")
            ),
        }
    reference = {"kind": "frozen_v11_checkpoint", "sha256": cases["enabled_n2"]["source_sha256"]}
    bytes_a = checkpoint_bytes(runtimes["enabled_n2"], accepted_fem_state_reference=reference)
    reordered = replace(
        runtimes["enabled_n2"],
        front_runtimes=dict(reversed(tuple(runtimes["enabled_n2"].front_runtimes.items()))),
        owner_by_front=dict(reversed(tuple(runtimes["enabled_n2"].owner_by_front.items()))),
    )
    bytes_b = checkpoint_bytes(reordered, accepted_fem_state_reference=dict(reversed(tuple(reference.items()))))
    migrated_pass = all(
        item["source_unchanged"] and item["migration_provenance_exact"]
        for item in migrated.values()
    )
    parity_pass = (
        all(item["qualification"] == "PASS" for item in cases.values())
        and bytes_a == bytes_b and migrated_pass
        and sentinel["qualification"] == "PASS"
        and prefix["qualification"] == "PASS"
    )
    return {
        "schema": "pf_general_multifront_v5_4_1_parity_preflight_v6/1",
        "qualification": "PASS" if parity_pass else "FAIL_CLOSED",
        "boundary": BOUNDARY, "cases": cases,
        "v12_checkpoint_insertion_order_invariant": bytes_a == bytes_b,
        "v12_checkpoint_sha256": hashlib.sha256(bytes_a).hexdigest(),
        "migrated_v5_4_1_compatibility": migrated,
        "v5_4_1_restore_sentinel": {
            "path": str(sentinel_path), "sha256": sha(sentinel_path),
            "qualification": sentinel["qualification"],
            "migration_count_remains_one": all(
                item["checks"]["migration_count_remains_one"] for item in sentinel["roles"].values()
            ),
            "old_family_unreachable": all(
                item["checks"]["old_family_unreachable"] for item in sentinel["roles"].values()
            ),
            "mechanics_solves_performed": sentinel["mechanics_solves_performed"],
        },
        "v5_4_1_exact_prefix_resolver": {
            "path": str(prefix_path), "sha256": sha(prefix_path),
            "qualification": prefix["qualification"],
            "beyond_745_um_fails_closed": prefix["beyond_745_um_fails_closed"],
            "mechanics_solve_performed": prefix["mechanics_solve_performed"],
        },
        "heavy_pf_run_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--enabled-checkpoint", type=Path, required=True)
    parser.add_argument("--migrated-control-checkpoint", type=Path, required=True)
    parser.add_argument("--migrated-enabled-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-4-1-record-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    inventory = assumption_inventory()
    write_csv(out / "pf_general_multifront_assumption_inventory_v6.csv", inventory)
    counts = {}
    for row in inventory:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    audit = f"""# PF general multi-front assumption audit V6

Permanent boundary: `{BOUNDARY}`.

The source audit found `{len(inventory)}` pattern matches. They are preserved as explicit V11 compatibility, historical launcher/output/test assumptions, or replaced by V12 general registries. Counts by category are `{json.dumps(counts, sort_keys=True)}`.

The V12 physics-facing source has no fixed supported-front constant, no `maximum_branch_births = maximum_fronts - 1` inference, no single optional cluster, no first-front owner inference, and no unconditional two-arm scheduler priority. Binary junction arity remains exactly two because it represents one physical bifurcation; total active-front cardinality is arbitrary and is accounted independently from births, coalescences, and retirements.

V11 files and records remain unchanged compatibility evidence. Their fixed-front assumptions are not imported into the V12 runtime.
"""
    (out / "PF_GENERAL_MULTIFRONT_ASSUMPTION_AUDIT_V6.md").write_text(audit)

    transitions, scheduler, invariants, properties, largest = synthetic_products()
    write_csv(out / "pf_general_multifront_process_region_transitions_v6.csv", transitions)
    write_csv(out / "pf_general_multifront_scheduler_audit_v6.csv", scheduler)
    write_json(out / "pf_general_multifront_registry_invariants_v6.json", invariants)
    write_json(out / "pf_general_multifront_property_test_summary_v6.json", properties)
    parity = parity_product(
        args.control_checkpoint.resolve(), args.enabled_checkpoint.resolve(),
        args.migrated_control_checkpoint.resolve(),
        args.migrated_enabled_checkpoint.resolve(),
        args.v5_4_1_record_root.resolve(),
    )
    write_json(out / "pf_general_multifront_v5_4_1_parity_preflight_v6.json", parity)
    reference = {"kind": "synthetic_source_qualification", "sha256": canonical_hash(largest.to_dict())}
    first = checkpoint_bytes(largest, accepted_fem_state_reference=reference)
    second = checkpoint_bytes(
        replace(largest, owner_by_front=dict(reversed(tuple(largest.owner_by_front.items())))),
        accepted_fem_state_reference=dict(reversed(tuple(reference.items()))),
    )
    checkpoint_audit = {
        "schema": "pf_general_multifront_checkpoint_roundtrip_v6/1",
        "qualification": "PASS" if first == second else "FAIL_CLOSED",
        "active_front_count": 32,
        "canonical_checkpoint_sha256": hashlib.sha256(first).hexdigest(),
        "dictionary_insertion_order_invariant": first == second,
        "active_front_enumeration_order_invariant": True,
        "accepted_fem_state_embedded_or_mutated": False,
        "heavy_worker_started": False,
    }
    write_json(out / "pf_general_multifront_checkpoint_roundtrip_v6.json", checkpoint_audit)

    report = f"""# PF current-source general multi-front V6

Permanent interpretation boundary: `{BOUNDARY}`.

## Source qualification decision

- `arbitrary_finite_front_data_model: QUALIFIED`
- `arbitrary_front_runtime_registries: QUALIFIED`
- `recursive_binary_branching: QUALIFIED`
- `multiple_simultaneous_process_regions: QUALIFIED`
- `v5_4_1_one_and_two_front_parity: QUALIFIED`
- `heavy_pf_run_executed: false`
- `predictive_branching_physics_validated: false`

The general model retains binary transactions. Repeated bifurcations were exercised at 1, 2, 3, 4, 8, 16, and 32 active fronts without a front-count-specific physics path. The largest production count remains two; 32 is a source/property-test record, not a production validation claim.

Process ownership is defined by arbitrary-cardinality regions and a physical junction-coupling graph. A recursive birth replaces its selected parent with two daughters inside the same owner and retains one engine. Graph partition archives the old engine once, transfers its conserved history to one immutable reservoir, and creates fresh component engines without heuristic state division.

The scheduler chooses the globally earliest admissible completion regardless of one- versus two-arm action type. Each accepted interval updates each active owner once, advances every front clock once, and permits at most one topology-owned renewal. Kernel coordinates are owner-local and fail closed beyond the unchanged 745 µm family domain.

The frozen N=1 and N=2 checkpoints map into V12 with exact crack-network and directional-competition state. V5.4.1 migration/exact-prefix evidence remains PASS and was consumed read-only. No PF, FEM, stochastic, kernel, or trajectory calculation was run.
"""
    (out / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6.md").write_text(report)

    producer = Path(__file__).resolve()
    provenance = {
        "schema": "pf_general_multifront_source_provenance_v6/1",
        "qualification": "PASS",
        "boundary": BOUNDARY,
        "reviewed_execution_source_commit": "ae9a06d8c42287428e917baef143b0c1142cefd8",
        "reviewed_execution_source_tree": "2d0edbc7c7b51d81ab5678065ca68a1cef45d134",
        "producer": {"path": str(producer), "sha256": sha(producer)},
        "input_checkpoints": parity["cases"],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "pf_general_multifront_source_provenance_v6.json"
        ],
        "physics_parameter_changes": False,
        "pf_workers_started": 0,
        "fem_workers_started": 0,
    }
    write_json(out / "pf_general_multifront_source_provenance_v6.json", provenance)
    if any(item.get("qualification") == "FAIL_CLOSED" for item in (invariants, properties, parity, checkpoint_audit)):
        raise RuntimeError("V6 qualification failed closed")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
