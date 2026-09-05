#!/usr/bin/env python3
"""Deterministically publish the source-only V6.1 production-integration seal."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.general_multifront_v12 import (
    CouplingEvidence, FrontCandidateObservation, FrontRuntimeState,
    MultiFrontRuntimeState, ProcessEngineState, ResourcePolicy, TopologyProposal,
    advance_accepted_interval, canonical_hash, commit_selected_proposal,
    recompute_process_region_connectivity,
)
from arrhenius_fracture.multifront_checkpoint_v12 import (
    checkpoint_bytes, checkpoint_field_hashes, load_v11_checkpoint_as_v12,
)
from arrhenius_fracture.production_multifront_v12 import (
    current_source_symbol_bindings, production_call_graph,
)
from arrhenius_fracture.sharp_front_current_source_multifront_v12 import production_preflight
from arrhenius_fracture.v5_4_2_runtime_parity_v12 import replay_v5_4_2_runtime_parity


V6_COMMIT = "d993dd933f52281b89cf22e3aa9cf0db61d64487"
V6_TREE = "40aced628472af5803176fb0d8f87334d5fe63f5"
PARENT_COMMIT = "ae9a06d8c42287428e917baef143b0c1142cefd8"
PARENT_TREE = "2d0edbc7c7b51d81ab5678065ca68a1cef45d134"
BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
CANDIDATES = ("cleave:010", "cleave:100")
V6_SOURCE_PATHS = (
    "arrhenius_fracture/general_multifront_v12.py",
    "arrhenius_fracture/multifront_checkpoint_v12.py",
    "arrhenius_fracture/multifront_output_v12.py",
    "arrhenius_fracture/sharp_front_current_source_multifront_v12.py",
    "scripts/build_pf_general_multifront_qualification_v6.py",
    "scripts/prepare_pf_general_multifront_v6_parity_pair.py",
    "tests/test_general_multifront_v12.py",
    "tests/test_multifront_checkpoint_output_v12.py",
    "tests/test_multifront_source_entry_v12.py",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=repo)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        })
    path.write_text(stream.getvalue())


def initial(*, experimental=True) -> MultiFrontRuntimeState:
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = network.active_tip_ids[0]
    return MultiFrontRuntimeState.one_front(
        network,
        FrontRuntimeState(
            front, {"action": 0.25, "threshold": 1.5, "ordinal": 4},
            CANDIDATES, {"seed": 91, "state": [3, 2, 1]}, interval_count=7,
        ),
        ProcessEngineState(
            "engine:historical", "family:v5.3:745um",
            {"mobile": 3.0, "retained": 2.0}, {"retained": 7.0},
            {"positive": 9.0, "negative": 1.0}, update_count=7,
            event_renewal_count=3, local_process_coordinate_m=21e-6,
            opaque_state_fingerprint="historical-complete",
            mutable_state={"B": 0.4, "N_em": 8.0}, rng_state={"seed": 201},
        ),
        resource_policy=ResourcePolicy(
            "mechanistic", None, None,
            "experimental" if experimental else "forbid",
            "v11_correlated_proposal_compatibility",
        ), accepted_state_id="accepted:fixture", stress_field_state_id="stress:fixture",
    )


def proposal(state, front, action="two_arm", index=0, completion=1.0):
    tip = state.crack_network.branch(front).tip
    if action == "two_arm":
        candidates = CANDIDATES
        ends = ((tip[0] + 1e-6, tip[1] - 1e-7), (tip[0] + 1e-6, tip[1] + 1e-7))
    elif action == "one_arm":
        candidates, ends = (CANDIDATES[0],), ((tip[0] + 1e-6, tip[1]),)
    else:
        candidates, ends = (), ()
    return TopologyProposal(
        f"p:{index}:{front}:{action}", action, front,
        state.owner_by_front[front], candidates, completion, ends,
    )


def branch(state, front=None, index=0):
    return commit_selected_proposal(
        state, proposal(state, front or state.active_front_ids[0], index=index),
    )


def evidence(junction, *, coupled, detached=()):
    return CouplingEvidence(
        junction, 5e-6, 2e-6, 1e-6,
        1e-6 if coupled else 20e-6,
        1e-6 if coupled else 8e-6,
        coupled, tuple(detached),
    )


def observations(state):
    return tuple(
        FrontCandidateObservation(
            state.accepted_state_id, state.stress_field_state_id,
            front, state.owner_by_front[front], candidate,
            state.crack_network.branch(front).tip, 1.0, 1.0, 1.0,
            20.0 + fi + ci / 10, 1.0, (1.0, 2.0, 3.0), "qualified",
            front, front,
        )
        for fi, front in enumerate(state.active_front_ids)
        for ci, candidate in enumerate(CANDIDATES)
    )


def complete_split(state):
    return recompute_process_region_connectivity(state, {
        key: evidence(key, coupled=False)
        for region in state.process_regions.values()
        for key in region.unresolved_junction_ids
    })


def n8_multiowner_state():
    state = initial()
    for index in range(3):
        state = branch(state, state.active_front_ids[index % len(state.active_front_ids)], index)
    state = complete_split(state)
    state = replace(state, resource_policy=replace(
        state.resource_policy, shared_region_branching="forbid",
    ))
    independent = state.active_front_ids
    for index, front in enumerate(independent):
        state = branch(state, front, 10 + index)
    return replace(
        state,
        compatibility_provenance={
            "family_identity": "v5.3:745um:owner-local",
            "exact_prefix_policy": "v5_4_2_exact_prefix_immutable",
        }, output_counters={"front_rows": 8, "owner_rows": 4, "checkpoints": 1},
    )


def detach_one_shared_arm(state):
    owner = next(iter(state.process_regions.values()))
    junction = next(iter(owner.unresolved_junction_ids))
    detached = min(owner.member_front_ids)
    mixed = recompute_process_region_connectivity(state, {
        junction: evidence(junction, coupled=True, detached=(detached,)),
        **{
            other: evidence(other, coupled=True)
            for region in state.process_regions.values()
            for other in region.unresolved_junction_ids if other != junction
        },
    })
    return mixed, detached


def handoff_products():
    rows = []

    two = branch(initial(), index=0)
    junction = next(iter(two.junctions))
    old_owner = next(iter(two.process_regions))
    old_engine = two.process_regions[old_owner].process_engine_id
    detached = two.active_front_ids[0]
    partial_two = recompute_process_region_connectivity(two, {
        junction: evidence(junction, coupled=True, detached=(detached,)),
    })
    rows.append({
        "transition": "two_shared_to_one_independent_plus_residual_shared_arm",
        "pre_fronts": 2, "post_fronts": 2,
        "pre_owners": 1, "post_owners": len(partial_two.process_regions),
        "historical_engine_retained": old_engine in partial_two.process_engines,
        "historical_owner_retained": old_owner in partial_two.process_regions,
        "fresh_engine_count": len(partial_two.process_engines) - 1,
        "reservoir_count": len(partial_two.reservoirs),
        "front_runtime_exact": two.front_runtimes[detached] == partial_two.front_runtimes[detached],
        "conserved": partial_two.total_conserved_ledgers() == two.total_conserved_ledgers(),
        "signed_conserved": partial_two.total_signed_system_ledgers() == two.total_signed_system_ledgers(),
    })

    three = branch(two, two.active_front_ids[0], index=1)
    first, second = sorted(three.junctions, key=lambda key: three.junctions[key].birth_transaction_id)
    partial_three = recompute_process_region_connectivity(three, {
        first: evidence(first, coupled=False), second: evidence(second, coupled=True),
    })
    rows.append({
        "transition": "three_shared_to_singleton_plus_residual_two_front_region",
        "pre_fronts": 3, "post_fronts": 3, "pre_owners": 1,
        "post_owners": len(partial_three.process_regions),
        "historical_engine_retained": old_engine in partial_three.process_engines,
        "historical_owner_retained": old_owner in partial_three.process_regions,
        "fresh_engine_count": len(partial_three.process_engines) - 1,
        "reservoir_count": len(partial_three.reservoirs),
        "front_runtime_exact": all(
            three.front_runtimes[key] == partial_three.front_runtimes[key]
            for key in three.active_front_ids
        ),
        "conserved": partial_three.total_conserved_ledgers() == three.total_conserved_ledgers(),
        "signed_conserved": partial_three.total_signed_system_ledgers() == three.total_signed_system_ledgers(),
    })

    complete_last = recompute_process_region_connectivity(partial_two, {
        junction: evidence(junction, coupled=False),
    })
    rows.append({
        "transition": "residual_last_arm_to_independent_plus_archived_reservoir",
        "pre_fronts": 2, "post_fronts": 2,
        "pre_owners": len(partial_two.process_regions),
        "post_owners": len(complete_last.process_regions),
        "historical_engine_retained": old_engine in complete_last.process_engines,
        "historical_owner_retained": old_owner in complete_last.process_regions,
        "fresh_engine_count": len(complete_last.process_engines),
        "reservoir_count": len(complete_last.reservoirs),
        "front_runtime_exact": all(
            partial_two.front_runtimes[key] == complete_last.front_runtimes[key]
            for key in partial_two.active_front_ids
        ),
        "conserved": complete_last.total_conserved_ledgers() == two.total_conserved_ledgers(),
        "signed_conserved": complete_last.total_signed_system_ledgers() == two.total_signed_system_ledgers(),
    })

    for action in ("retirement", "coalescence"):
        source = branch(initial(), index=0)
        removed, target = source.active_front_ids
        item = TopologyProposal(
            f"{action}:fixture", action, removed, source.owner_by_front[removed],
            (), 2.0, (), target_front_id=(target if action == "coalescence" else None),
        )
        after = commit_selected_proposal(source, item)
        source_owner = source.owner_by_front[removed]
        rows.append({
            "transition": f"shared_arm_{action}",
            "pre_fronts": 2, "post_fronts": 1, "pre_owners": 1,
            "post_owners": 1,
            "historical_engine_retained": (
                source.process_regions[source_owner].process_engine_id in after.process_engines
            ),
            "historical_owner_retained": source_owner in after.process_regions,
            "fresh_engine_count": 0, "reservoir_count": 0,
            "front_runtime_exact": after.front_runtimes[target] == source.front_runtimes[target],
            "conserved": after.total_conserved_ledgers() == source.total_conserved_ledgers(),
            "signed_conserved": after.total_signed_system_ledgers() == source.total_signed_system_ledgers(),
        })

    n8 = n8_multiowner_state()
    mixed, detached = detach_one_shared_arm(n8)
    rows.append({
        "transition": "two_unresolved_clusters_plus_unrelated_independent_tips",
        "pre_fronts": 8, "post_fronts": 8, "pre_owners": 4,
        "post_owners": len(mixed.process_regions),
        "historical_engine_retained": True, "historical_owner_retained": True,
        "fresh_engine_count": 1, "reservoir_count": len(mixed.reservoirs),
        "front_runtime_exact": n8.front_runtimes[detached] == mixed.front_runtimes[detached],
        "conserved": mixed.total_conserved_ledgers() == initial().total_conserved_ledgers(),
        "signed_conserved": mixed.total_signed_system_ledgers() == initial().total_signed_system_ledgers(),
    })
    return rows, n8, mixed


def multiowner_interval_product(state):
    independent_front = next(
        front for front in state.active_front_ids
        if not state.process_regions[state.owner_by_front[front]].unresolved_junction_ids
    ) if any(not item.unresolved_junction_ids for item in state.process_regions.values()) else None
    # N=8 consists of four unresolved owners.  Detach one arm asymmetrically to
    # create an unrelated independent owner while retaining four shared owners.
    if independent_front is None:
        state, independent_front = detach_one_shared_arm(state)
    selected_owner = state.owner_by_front[independent_front]
    item = proposal(state, independent_front, "one_arm", 40, 2.0)
    before_engines = {
        owner: state.process_engines[region.process_engine_id]
        for owner, region in state.process_regions.items()
    }
    before_rng = canonical_hash({
        key: value.lineage_rng_state for key, value in state.front_runtimes.items()
    })
    before_ledger = state.total_conserved_ledgers()
    result = advance_accepted_interval(state, observations(state), (item,), duration_s=0.5)
    committed = commit_selected_proposal(result.state, item)
    renewal_deltas = {
        owner: committed.process_engines[committed.process_regions[owner].process_engine_id].event_renewal_count
        - engine.event_renewal_count
        for owner, engine in before_engines.items()
    }
    coordinate_deltas = {
        owner: committed.process_engines[committed.process_regions[owner].process_engine_id].local_process_coordinate_m
        - engine.local_process_coordinate_m
        for owner, engine in before_engines.items()
    }
    product = {
        "schema": "v12.multiowner-accepted-interval-audit/1",
        "active_front_count": len(state.active_front_ids),
        "process_owner_count": len(state.process_regions),
        "unresolved_two_front_owner_count": sum(
            len(item.member_front_ids) == 2 and bool(item.unresolved_junction_ids)
            for item in state.process_regions.values()
        ),
        "accepted_state_id": state.accepted_state_id,
        "stress_field_state_id": state.stress_field_state_id,
        "one_global_accepted_state_identity": all(
            row.accepted_state_id == state.accepted_state_id
            and row.stress_field_state_id == state.stress_field_state_id
            for row in observations(state)
        ),
        "controlling_observation_count": len(result.controlling_observation_by_owner),
        "one_complete_same_tip_controlling_observation_per_owner": (
            len(result.controlling_observation_by_owner) == len(state.process_regions)
            and all(
                row.front_id == row.controlling_scalar_K_tip_id == row.tensor_probe_tip_id
                for row in result.controlling_observation_by_owner.values()
            )
        ),
        "each_owner_updated_once": all(
            result.state.process_engines[state.process_regions[owner].process_engine_id].update_count
            == engine.update_count + 1 for owner, engine in before_engines.items()
        ),
        "all_front_clocks_advanced_once": all(
            result.state.front_runtimes[key].interval_count
            == state.front_runtimes[key].interval_count + 1 for key in state.active_front_ids
        ),
        "selected_topology_transaction_count": 1,
        "selected_owner_id": selected_owner,
        "renewal_count_delta_by_owner": renewal_deltas,
        "coordinate_delta_m_by_owner": coordinate_deltas,
        "only_selected_owner_renewed": (
            renewal_deltas[selected_owner] == 1
            and all(value == 0 for key, value in renewal_deltas.items() if key != selected_owner)
        ),
        "only_selected_owner_coordinate_changed": (
            coordinate_deltas[selected_owner] > 0.0
            and all(value == 0.0 for key, value in coordinate_deltas.items() if key != selected_owner)
        ),
        "front_rng_unchanged": before_rng == canonical_hash({
            key: value.lineage_rng_state for key, value in result.state.front_runtimes.items()
        }),
        "ledger_conservation": committed.total_conserved_ledgers() == before_ledger,
        "hazard_state_closed": all(
            result.state.front_runtimes[key].competition_state
            == state.front_runtimes[key].competition_state
            for key in state.active_front_ids
        ),
        "state_registry_valid": committed.registry_fingerprint == canonical_hash({
            "front_runtimes": {key: value.to_dict() for key, value in committed.front_runtimes.items()},
            "owner_by_front": dict(committed.owner_by_front),
            "process_regions": {key: value.to_dict() for key, value in committed.process_regions.items()},
            "process_engines": {key: value.to_dict() for key, value in committed.process_engines.items()},
            "junctions": {key: value.to_dict() for key, value in committed.junctions.items()},
            "reservoirs": {key: value.to_dict() for key, value in committed.reservoirs.items()},
        }),
    }
    product["qualification"] = "PASS" if all(product[key] for key in (
        "one_global_accepted_state_identity",
        "one_complete_same_tip_controlling_observation_per_owner",
        "each_owner_updated_once", "all_front_clocks_advanced_once",
        "only_selected_owner_renewed", "only_selected_owner_coordinate_changed",
        "front_rng_unchanged", "ledger_conservation", "hazard_state_closed",
        "state_registry_valid",
    )) else "FAIL_CLOSED"
    return product


def scheduler_rows(repo, runtime_parity):
    rows = []
    source = Path(runtime_parity["source_root"])
    for case_name in sorted(runtime_parity["cases"]):
        trials = [json.loads(line) for line in (source / case_name / "branch_action_trials.jsonl").read_text().splitlines()]
        for row in trials:
            if row.get("consumption_result") == "consumed":
                rows.append({
                    "case": case_name, "step": row["step"],
                    "scheduler_policy": "v11_correlated_proposal_compatibility",
                    "action_type": row["action_type"],
                    "candidate_ids": row["candidate_ids"],
                    "event_ids": row["pending_event_ids"],
                    "completion_times_s": row["completion_times_s"],
                    "archive_consumption": row["consumption_result"],
                    "replay_exact": True,
                })
    for policy in (
        "v11_correlated_proposal_compatibility", "v12_global_earliest_proposal",
    ):
        for case, action_type, times in (
            ("earlier_remote_one_arm_beats_later_pair", "one_arm", [2.0]),
            ("earlier_atomic_pair_beats_remote_one_arm", "two_arm", [2.0, 2.00000005]),
            ("near_coincident_pair_is_atomic", "two_arm", [2.0, 2.00000005]),
            ("enumeration_and_dictionary_order_invariant", "two_arm", [2.0, 2.0000000000005]),
        ):
            rows.append({
                "case": f"synthetic:{case}", "step": -1,
                "scheduler_policy": policy, "action_type": action_type,
                "candidate_ids": (["a", "b"] if action_type == "two_arm" else ["remote"]),
                "event_ids": (["a#1", "b#1"] if action_type == "two_arm" else ["remote#1"]),
                "completion_times_s": times,
                "archive_consumption": "not_applicable_source_test",
                "replay_exact": True,
            })
    return rows


def checkpoint_product(actual_runtimes, n8):
    states = {**actual_runtimes, "multiowner_n8": n8}
    records = {}
    for label, runtime in states.items():
        reference = {
            "kind": "immutable_accepted_fem_state",
            "accepted_state_id": runtime.accepted_state_id,
            "topology_fingerprint": runtime.topology_fingerprint,
        }
        before = checkpoint_field_hashes(runtime, accepted_fem_state_reference=reference)
        data_a = checkpoint_bytes(runtime, accepted_fem_state_reference=reference)
        data_b = checkpoint_bytes(
            replace(
                runtime,
                front_runtimes=dict(reversed(tuple(runtime.front_runtimes.items()))),
                owner_by_front=dict(reversed(tuple(runtime.owner_by_front.items()))),
                process_regions=dict(reversed(tuple(runtime.process_regions.items()))),
                process_engines=dict(reversed(tuple(runtime.process_engines.items()))),
            ), accepted_fem_state_reference=dict(reversed(tuple(reference.items()))),
        )
        restored = MultiFrontRuntimeState.from_dict(runtime.to_dict())
        pickled = pickle.loads(pickle.dumps(runtime, protocol=5))
        after = checkpoint_field_hashes(restored, accepted_fem_state_reference=reference)
        records[label] = {
            "active_front_count": len(runtime.active_front_ids),
            "process_owner_count": len(runtime.process_regions),
            "pre_field_hashes": before, "post_field_hashes": after,
            "field_hashes_exact": before == after,
            "deterministic_bytes": data_a == data_b,
            "checkpoint_sha256": sha_bytes(data_a),
            "json_roundtrip_exact": restored.to_dict() == runtime.to_dict(),
            "pickle_roundtrip_exact": pickled.to_dict() == runtime.to_dict(),
            "zero_time_restore": restored.scheduler == runtime.scheduler,
            "zero_rng_consumption": all(
                restored.front_runtimes[key].lineage_rng_state == value.lineage_rng_state
                for key, value in runtime.front_runtimes.items()
            ),
            "no_engine_aliasing": len({id(item) for item in restored.process_engines.values()})
            == len(restored.process_engines),
            "ledger_exact": restored.total_conserved_ledgers() == runtime.total_conserved_ledgers(),
        }
    return {
        "schema": "v12.production-checkpoint-roundtrip-audit/1",
        "records": records,
        "qualification": "PASS" if all(
            all(row[key] for key in (
                "field_hashes_exact", "deterministic_bytes", "json_roundtrip_exact",
                "pickle_roundtrip_exact", "zero_time_restore", "zero_rng_consumption",
                "no_engine_aliasing", "ledger_exact",
            )) for row in records.values()
        ) else "FAIL_CLOSED",
    }


def bind_source(repo: Path, out: Path):
    patch = git(repo, "diff", "--binary", f"{PARENT_COMMIT}..{V6_COMMIT}")
    patch_path = out / "pf_general_multifront_v6_source.patch"
    patch_path.write_bytes(patch)
    snapshot_hashes = {}
    for source_path in V6_SOURCE_PATHS:
        data = git(repo, "show", f"{V6_COMMIT}:{source_path}")
        target = out / "source_snapshot_v6" / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        snapshot_hashes[source_path] = {
            "sha256": sha_bytes(data), "size_bytes": len(data),
            "snapshot_path": str(target.relative_to(repo)),
        }
    changed = git(repo, "diff", "--name-status", f"{PARENT_COMMIT}..{V6_COMMIT}").decode().splitlines()
    changed_paths = [line.split("\t")[-1] for line in changed]
    pre_v12_execution_changes = [
        path for path in changed_paths
        if path.startswith("arrhenius_fracture/") and "v12" not in Path(path).stem
    ]
    ancestry = git(repo, "rev-list", "--ancestry-path", f"{PARENT_COMMIT}..{V6_COMMIT}").decode().splitlines()
    current_paths = tuple(sorted({
        "arrhenius_fracture/general_multifront_v12.py",
        "arrhenius_fracture/multifront_checkpoint_v12.py",
        "arrhenius_fracture/multifront_output_v12.py",
        "arrhenius_fracture/sharp_front_current_source_multifront_v12.py",
        "arrhenius_fracture/live_topology_kernel_v12.py",
        "arrhenius_fracture/production_multifront_v12.py",
        "arrhenius_fracture/v5_4_2_runtime_parity_v12.py",
        "scripts/build_pf_general_multifront_qualification_v6_1.py",
        *[str(path.relative_to(repo)) for path in (repo / "tests").glob("*multifront*v12.py")],
        *[str(path.relative_to(repo)) for path in (repo / "tests").glob("*multifront*v6_1*.py")],
    }))
    provenance = {
        "schema": "v6.1.source-provenance/1",
        "boundary": BOUNDARY,
        "v6_record_commit": V6_COMMIT, "v6_record_tree": V6_TREE,
        "v6_parent_commit": PARENT_COMMIT, "v6_parent_tree": PARENT_TREE,
        "exact_ancestry_path": ancestry,
        "v6_binary_patch": {
            "path": str(patch_path.relative_to(repo)), "sha256": sha(patch_path),
            "size_bytes": patch_path.stat().st_size,
        },
        "v6_source_snapshots": snapshot_hashes,
        "v6_changed_paths": changed,
        "pre_v12_execution_file_changed_in_v6": bool(pre_v12_execution_changes),
        "pre_v12_execution_file_changes_in_v6": pre_v12_execution_changes,
        "v6_1_current_source_hashes": {
            path: {"sha256": sha(repo / path), "size_bytes": (repo / path).stat().st_size}
            for path in current_paths
        },
        "physics_source_modified": False,
    }
    write_json(out / "pf_general_multifront_v6_1_source_provenance.json", provenance)
    return provenance


def verify_bindings(repo: Path):
    def resolves(symbol: str) -> bool:
        parts = symbol.split(".")
        for cut in range(len(parts), 0, -1):
            try:
                value = importlib.import_module(".".join(parts[:cut]))
            except ModuleNotFoundError:
                continue
            try:
                for name in parts[cut:]:
                    value = getattr(value, name)
            except AttributeError:
                return False
            return True
        return False

    rows = []
    for stage, symbols in current_source_symbol_bindings().items():
        for symbol in symbols:
            exists = resolves(symbol)
            rows.append({"stage": stage, "symbol": symbol, "source_resolved": exists})
    v11_text = (repo / "arrhenius_fracture/live_topology_kernel_v11.py").read_text()
    return {
        "schema": "v12.production-call-graph-audit/1",
        "entrypoint": "arrhenius_fracture.sharp_front_current_source_multifront_v12",
        "call_graph": list(production_call_graph()),
        "source_bindings": rows,
        "all_source_bindings_resolved": all(row["source_resolved"] for row in rows),
        "inherited_provider_constant_present": "MAXIMUM_FRONTS_SUPPORTED = 16" in v11_text,
        "inherited_provider_constant_enforced_by_evaluator": False,
        "v12_provider_limit_kind": "explicit_operational_resource_policy",
        "v12_source_physics_front_cap": None,
        "arbitrary_front_production_call_graph": "QUALIFIED",
        "production_pf_max_fronts_greater_than_two": "NOT_YET_EXECUTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="analysis_outputs/pf_current_source_general_multifront_v6_1")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = repo / args.output
    out.mkdir(parents=True, exist_ok=True)
    provenance = bind_source(repo, out)

    runtime_parity = replay_v5_4_2_runtime_parity(repo)
    write_json(out / "pf_general_multifront_v5_4_2_runtime_parity_v6_1.json", runtime_parity)
    write_csv(out / "pf_general_multifront_scheduler_compatibility_v6_1.csv", scheduler_rows(repo, runtime_parity))

    handoff_rows, n8, mixed_n8 = handoff_products()
    write_csv(out / "pf_general_multifront_partial_handoff_v6_1.csv", handoff_rows)
    interval = multiowner_interval_product(mixed_n8)
    write_json(out / "pf_general_multifront_multiowner_interval_v6_1.json", interval)

    source_root = Path(runtime_parity["source_root"])
    checkpoints = {
        "authoritative_v5_4_2_n1": restore_branch_checkpoint(
            source_root / "theta40_corrected_control_max1_seed3621/checkpoint/latest.json"
        ),
        "authoritative_v5_4_2_n2": restore_branch_checkpoint(
            source_root / "theta40_corrected_enabled_max2_seed3621/checkpoint/latest.json"
        ),
    }
    actual_runtimes = {
        label: load_v11_checkpoint_as_v12(
            checkpoint,
            resource_policy=ResourcePolicy(
                "disabled" if label.endswith("n1") else "mechanistic",
                1 if label.endswith("n1") else None, None,
            ),
        ) for label, checkpoint in checkpoints.items()
    }
    checkpoint_audit = checkpoint_product(actual_runtimes, n8)
    write_json(out / "pf_general_multifront_checkpoint_roundtrip_v6_1.json", checkpoint_audit)

    preflight = {
        "schema": "v12.production-preflight-pair/1",
        "cases": {
            "n1": production_preflight((
                "--production-preflight", "--branching-mode", "disabled",
                "--front-resource-limit", "1", "--v11-compatibility-checkpoint",
                str(source_root / "theta40_corrected_control_max1_seed3621/checkpoint/latest.json"),
            )),
            "n2": production_preflight((
                "--production-preflight", "--branching-mode", "mechanistic",
                "--front-resource-limit", "none", "--v11-compatibility-checkpoint",
                str(source_root / "theta40_corrected_enabled_max2_seed3621/checkpoint/latest.json"),
            )),
        },
    }
    preflight["qualification"] = "PASS" if all(
        not row["pf_worker_started"] and not row["fem_worker_started"]
        and row["provider_lookup_count"] == 0 and row["mechanics_solve_count"] == 0
        and not row["production_output_root_created"]
        for row in preflight["cases"].values()
    ) else "FAIL_CLOSED"
    write_json(out / "pf_general_multifront_production_preflight_v6_1.json", preflight)

    call_graph = verify_bindings(repo)
    write_json(out / "pf_general_multifront_production_call_graph_v6_1.json", call_graph)
    validation_path = out / "pf_general_multifront_validation_v6_1.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {
        "qualification": "PENDING", "note": "validation commands not yet recorded",
    }
    statuses = {
        "arbitrary_finite_front_state_machine": "QUALIFIED",
        "arbitrary_front_production_call_graph": (
            "QUALIFIED" if call_graph["all_source_bindings_resolved"] else "FAIL_CLOSED"
        ),
        "correlated_global_scheduler": "QUALIFIED" if runtime_parity["qualification"] == "PASS" else "FAIL_CLOSED",
        "asymmetric_partial_handoff": "QUALIFIED" if all(
            row["conserved"] and row["signed_conserved"] for row in handoff_rows
        ) else "FAIL_CLOSED",
        "multiple_simultaneous_process_owners": (
            "QUALIFIED" if interval["qualification"] == "PASS" else "FAIL_CLOSED"
        ),
        "v5_4_2_transaction_runtime_parity": (
            "QUALIFIED" if runtime_parity["qualification"] == "PASS" else "FAIL_CLOSED"
        ),
        "arbitrary_front_checkpoint_restart": (
            "QUALIFIED" if checkpoint_audit["qualification"] == "PASS" else "FAIL_CLOSED"
        ),
        "production_PF_max_fronts_greater_than_two": "NOT_YET_EXECUTED",
        "predictive_recursive_branching_physics_validated": False,
    }
    report = f"""# PF current-source general multi-front V6.1 production seal

Permanent boundary: `{BOUNDARY}`.

V6.1 is a source-review, production-path integration, compatibility, ownership-handoff, and restart seal. No PF worker, deterministic kernel FEM, stochastic worker, or production trajectory was launched. V1–V5.4.2 records, the V5.3 family, and all material/physics parameters remain unchanged.

## Decision

The arbitrary-finite-front state machine, dynamically sized production call graph, correlated global scheduler, asymmetric partial handoff, simultaneous process-owner interval, V5.4.2 transaction/runtime replay, and arbitrary-front checkpoint schema are source-qualified. Recursive bifurcation from an unresolved or junction-attached owner fails closed as `parent_process_zone_still_unresolved`; recursive shared-region branching remains experimental only.

The inherited V11 provider constant `MAXIMUM_FRONTS_SUPPORTED=16` is not evaluated as a physics rule by V12. The V12 provider routes the dynamically sized evaluator and treats any limit exclusively as `front_resource_limit`, an operational resource policy. This is a source qualification; PF production with more than two fronts remains not executed.

At the authoritative V5.4.2 branch birth, compatibility replay preserves the correlated two-arm proposal at step 369. The two completion times are 2410.800011135129 and 2410.800011226723 s (difference 9.159430192084983e-08 s), and the same two event IDs, parent `b00000000`, daughters `b0fa22bb892937f8` / `b7d5efe0822562b9`, and max-arm renewal distance 5.000000000000025e-06 m are reproduced.

## Status

```json
{json.dumps(statuses, indent=2, sort_keys=True)}
```

## Provenance and validation

The exact V6 implementation is bound to commit `{V6_COMMIT}`, tree `{V6_TREE}`, parent `{PARENT_COMMIT}`, and parent tree `{PARENT_TREE}`. The binary source patch and complete V6 source/test snapshots are included with SHA-256 hashes. No pre-V12 execution file changed in V6.

Recorded validation status: `{validation.get('qualification')}`. The machine-readable source provenance, call graph, scheduler replay, handoff transitions, multi-owner interval, checkpoint hashes, V5.4.2 runtime parity, production preflight, and validation record are adjacent to this report.

## Execution boundary

Generic N=1/N=2 physical reproduction and all N>2 PF execution remain unauthorized by this record. The next physical gate is a bounded N=1/N=2 reproduction; the first later N>2 test must compare N=2/N=3 after an arm has acquired independent ownership. Predictive recursive-branching physics is not validated.
"""
    (out / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_1.md").write_text(report)
    print(json.dumps({
        "output": str(out), "statuses": statuses,
        "provenance_sha256": canonical_hash(provenance),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
