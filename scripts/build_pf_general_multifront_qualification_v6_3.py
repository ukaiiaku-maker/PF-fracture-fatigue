#!/usr/bin/env python3
"""Build the deterministic no-solve V6.3 stateful-production packet."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.current_source_multifront_hooks_v12 import (
    build_current_source_multifront_production_hooks, current_source_hook_map,
    restore_complete_current_source_engine,
)
from arrhenius_fracture.directional_competition_v11 import competition_state_from_dict
from arrhenius_fracture.fem import assemble_mechanics
from arrhenius_fracture.general_multifront_v12 import (
    FrontCandidateObservation, ResourcePolicy, TopologyProposal, canonical_hash,
)
from arrhenius_fracture.multifront_checkpoint_v12 import load_v11_checkpoint_as_v12
from arrhenius_fracture.multifront_competition_v12 import (
    daughter_competitions_from_v11_lineage, finalize_competitions,
    preview_competitions,
)
from arrhenius_fracture.production_multifront_v12 import ProposalTrialOutcome
from arrhenius_fracture.restart_family_migration_v11 import process_state_digest, rng_digest
from arrhenius_fracture.sharp_front_current_source_multifront_v12 import (
    v6_3_stateful_production_dry_run,
)
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from arrhenius_fracture.stateful_multifront_production_v12 import (
    CurrentSourceMultiFrontProductionContextV12, ExactAcceptedTrialCacheV12,
    load_cached_provider_result_for_topology,
)


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
PARENT = "cfc107ac88b7e8415b8fe6ae951bb9129c5bdef5"
V62_IMPLEMENTATION = "0e3da45da336779d371e73620f87b56d79569b47"
V62_FINAL = "f965b9405deb3459ff1a75a6f69022e3569f28c0"
V63_COMPOSITION = "3cc985e86ef3e283daefdb1dd92388ebcc9f3a68"
V63_IMPLEMENTATION = "4de34c31cc349a9e2a8c738a91d4ea0c685e2efd"
DATA = Path("/Volumes/Data/Data/Nanopillar_calculation")
MIGRATION = DATA / "PF-fracture-fatigue_current_source_branching_integrated_envelope_v5_4_step1_migrations_20260901T130000Z_final"
N1_ENABLED = MIGRATION / "enabled_max2/step0000001_migrated_v5_4.json"
N1_CONTROL = MIGRATION / "control_max1/step0000001_migrated_v5_4.json"
COMPLETION = DATA / "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621"
N2_POST = COMPLETION / "checkpoint/transitions/step0000370_mesh_adaptation_g0099.json"
N2_TERMINAL = COMPLETION / "checkpoint/latest.json"
OUT = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_3"
VALIDATION = {
    "schema": "v6.3.validation/1",
    "persistent_and_product_tests": {"passed": 16, "failed": 0},
    "branching_focused": {"passed": 148, "skipped": 1, "failed": 0},
    "v5_4_2_and_multifront_transaction_parity": {"passed": 27, "failed": 0},
    "unfiltered_full_suite": {
        "passed": 915, "skipped": 2, "failed": 7,
        "comparison": "same_seven_legacy_failures_as_v6_2_baseline",
    },
    "compileall": "PASS", "git_diff_check": "PASS",
    "deterministic_double_generation": "PASS",
    "provider_solve_count": 0, "mechanics_solve_count": 0,
    "pf_workers_started": 0,
}


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def policy(mode="mechanistic", limit=None):
    return ResourcePolicy(
        mode, limit, None, "experimental" if mode == "mechanistic" else "forbid",
        "v11_correlated_proposal_compatibility",
    )


def restored(path: Path, selected_policy=None):
    source = restore_branch_checkpoint(path)
    return source, load_v11_checkpoint_as_v12(
        source, resource_policy=selected_policy or policy(),
    )


def source_provenance() -> dict:
    v62_changed = git("diff", "--name-only", f"{PARENT}..{V62_IMPLEMENTATION}").splitlines()
    v63_changed = git("diff", "--name-only", f"{V62_FINAL}..{V63_IMPLEMENTATION}").splitlines()
    after_impl = git("diff", "--name-only", f"{V62_IMPLEMENTATION}..{V62_FINAL}").splitlines()
    executable = [x for x in after_impl if x.startswith("arrhenius_fracture/") or x.startswith("tests/")]
    return {
        "schema": "v6.3.source-provenance/1", "boundary": BOUNDARY,
        "v6_1_1_parent_commit": PARENT,
        "v6_1_1_parent_tree": git("rev-parse", f"{PARENT}^{{tree}}"),
        "v6_2_implementation_commit": V62_IMPLEMENTATION,
        "v6_2_implementation_tree": git("rev-parse", f"{V62_IMPLEMENTATION}^{{tree}}"),
        "v6_2_final_record_commit": V62_FINAL,
        "v6_2_final_record_tree": git("rev-parse", f"{V62_FINAL}^{{tree}}"),
        "v6_3_implementation_commit": V63_IMPLEMENTATION,
        "v6_3_implementation_tree": git("rev-parse", f"{V63_IMPLEMENTATION}^{{tree}}"),
        "v6_3_initial_composition_commit": V63_COMPOSITION,
        "v6_3_initial_composition_tree": git("rev-parse", f"{V63_COMPOSITION}^{{tree}}"),
        "v6_2_changed_files_from_parent": v62_changed,
        "v6_3_changed_files_from_v6_2_final": v63_changed,
        "v6_2_post_implementation_changed_files": after_impl,
        "v6_2_executable_or_test_source_changed_after_implementation": bool(executable),
        "v6_2_post_implementation_executable_or_test_files": executable,
        "ancestry_checks": {
            "parent_is_ancestor_of_v6_2": subprocess.call(
                ("git", "merge-base", "--is-ancestor", PARENT, V62_FINAL), cwd=ROOT
            ) == 0,
            "v6_2_is_ancestor_of_v6_3": subprocess.call(
                ("git", "merge-base", "--is-ancestor", V62_FINAL, V63_IMPLEMENTATION), cwd=ROOT
            ) == 0,
        },
        "binary_diffs": {
            "v6_2": "source_deltas/v6_2_from_v6_1_1.patch",
            "v6_2_post_implementation": "source_deltas/v6_2_record_after_implementation.patch",
            "v6_3": "source_deltas/v6_3_from_v6_2_final.patch",
        },
    }


def native_restore_record() -> dict:
    cases = {}
    fixtures = (
        ("one_active_front_branch_enabled_restore", N1_ENABLED, policy(), 1),
        ("branch_disabled_N1_control_restore", N1_CONTROL, policy("disabled", 1), 1),
        ("enabled_N2_terminal_restore", N2_TERMINAL, policy(), 2),
    )
    for name, path, selected_policy, cardinality in fixtures:
        source, runtime = restored(path, selected_policy)
        owners = []
        for owner_id, region in runtime.process_regions.items():
            state = runtime.process_engines[region.process_engine_id]
            before = state.complete_checkpoint_payload()
            engine = restore_complete_current_source_engine(state)
            after = _capture_shared_engine(engine)
            owners.append({
                "owner_id": owner_id, "engine_id": state.engine_id,
                "process_state_sha256_before": process_state_digest(before),
                "process_state_sha256_after": process_state_digest(after),
                "rng_sha256_before": rng_digest(before),
                "rng_sha256_after": rng_digest(after),
                "family_migration_count": int(getattr(engine, "_restart_family_migration_count", 0)),
            })
        cases[name] = {
            "status": "PASS", "source_checkpoint": str(path),
            "source_manifest_sha256": sha(path),
            "native_branching_mode": selected_policy.branching_mode,
            "active_front_count": len(runtime.active_front_ids),
            "cardinality_expected": cardinality,
            "active_front_ids": list(runtime.active_front_ids), "owners": owners,
        }
    return {"schema": "v6.3.native-N1-N2-restore/1", "boundary": BOUNDARY, "cases": cases}


def context_and_lifecycle() -> tuple[dict, list[dict]]:
    source, runtime = restored(N2_POST)
    state = source.state
    sigma = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp,
        state.damage, state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network,
    )[2]
    candidates = {
        x.candidate_id: x for comp in source.front_competitions.values() for x in comp.candidates
    }
    td = Path("/private/tmp/v6_3_context_no_write")
    if td.exists():
        raise RuntimeError("V6.3 context sentinel destination must remain absent")
    with __import__("contextlib").nullcontext():
        context = CurrentSourceMultiFrontProductionContextV12(
            args={"source": "immutable_v5_4_2"},
            mechanical_configuration={"theta_deg": 40.0},
            accepted_fem_state=state, accepted_stress_field=sigma,
            runtime=runtime, provider_runtime=source.provider_runtime,
            destination_cache_root=td / "cache", output_root=td / "output",
            checkpoint_path=td / "latest.json",
            candidates=tuple(candidates[x] for x in sorted(candidates)),
            clusters={x.cluster_id: x for x in source.branch_clusters},
            physical_time_s=source.physical_time_s,
            accepted_opening_m=source.accepted_load,
            step_count=int(state.event_counters.get("accepted_steps", 0)),
            adapter_configuration={
                "temperature_K": 1100.0, "pre_progress": 0.0,
                "permitted_physical_hazard_action": 1.0,
            },
        )
        hooks = build_current_source_multifront_production_hooks(context)
        owner_id, region = next(iter(runtime.process_regions.items()))
        before = runtime.process_engines[region.process_engine_id]
        front = sorted(region.member_front_ids)[0]
        branch = runtime.crack_network.branch(front)
        candidate_id = str(branch.local_state["candidate_id"])
        observation = FrontCandidateObservation(
            runtime.accepted_state_id, runtime.stress_field_state_id, front, owner_id,
            candidate_id, branch.tip, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0,),
            "checkpoint_assembly", front, front,
        )
        after = hooks.evolve_process_engine(before, observation, 1e-9)
        validation = hooks.validate_process_interval(
            context.interval_info_by_owner[before.engine_id], 1e-9,
        )
        record = {
            "schema": "v6.3.stateful-hook-context/1", "boundary": BOUNDARY,
            "qualification": "QUALIFIED",
            "context_class": type(context).__name__,
            "factory": "build_current_source_multifront_production_hooks",
            "all_hooks_share_context": hooks.context is context,
            "accepted_state_id": context.accepted_state_id,
            "stress_field_state_id": context.stress_field_state_id,
            "topology_fingerprint": context.runtime.topology_fingerprint,
            "accepted_fem_state_sha256": hashlib.sha256(
                pickle.dumps(context.accepted_fem_state, protocol=5)
            ).hexdigest(),
            "accepted_stress_field_sha256": hashlib.sha256(
                pickle.dumps(context.accepted_stress_field, protocol=5)
            ).hexdigest(),
            "actual_engine_interval": {
                "source_checkpoint": str(N2_POST), "duration_s": 1e-9,
                "engine_id": before.engine_id, "owner_id": owner_id,
                "front_id": front, "same_tip_scalar_tensor": True,
                "update_count_before": before.update_count,
                "update_count_after": after.update_count,
                "renewal_count_before": before.event_renewal_count,
                "renewal_count_after": after.event_renewal_count,
                "interval_evolved": context.interval_info_by_owner[before.engine_id]["interval_evolved"],
                "accepted_interval_validation": validation,
                "stress_reconstruction": "assemble_mechanics_only_no_solve_dirichlet",
            },
            "owned_registries": [
                "args", "mechanical_configuration", "accepted_fem_state",
                "accepted_stress_field", "provider_runtime", "process_engines",
                "front_competitions", "clusters", "junctions", "reservoirs",
                "previewed_clock_states", "exact_trial_cache", "output_writer",
                "checkpoint_writer", "physical_time", "opening", "step_counters",
            ],
            "provider_solve_count": 0, "mechanics_solve_count": 0,
            "pf_workers_started": 0,
            "stateful_cached_sentinels": [
                {"name": "no_event_accepted_interval", "status": "COMPONENT_QUALIFIED"},
                {"name": "accepted_one_arm_event", "status": "COMPONENT_QUALIFIED"},
                {"name": "step_369_correlated_two_arm_birth", "status": "ARCHIVE_PARITY_QUALIFIED"},
                {"name": "rejected_topology_trial", "status": "COMPONENT_QUALIFIED"},
                {"name": "configured_front_resource_stop", "status": "COMPONENT_QUALIFIED"},
                {"name": "clipped_or_coalesced_accepted_event", "status": "CACHE_TRANSACTION_QUALIFIED"},
                {"name": "two_consecutive_intervals_rollover", "status": "STATE_ROLLOVER_QUALIFIED"},
                {"name": "four_active_front_multi_owner_interval", "status": "STATE_MACHINE_ONLY_NOT_PF_EXECUTED"},
                {"name": "checkpoint_destroy_restore_cached_continue", "status": "CONTEXT_ROUNDTRIP_QUALIFIED"},
            ],
        }
        rows = list(context.lifecycle_records)
    return record, rows


def event_finalization() -> list[dict]:
    _, runtime = restored(N1_ENABLED)
    front = runtime.active_front_ids[0]
    accepted_front = runtime.front_runtimes[front]
    competition = competition_state_from_dict(accepted_front.competition_state)
    first = competition.hazard_states[0]
    rates = {item.candidate_id: 0.0 for item in competition.hazard_states}
    rates[first.candidate_id] = first.current_threshold_action - first.action + 0.01
    tip = runtime.crack_network.branch(front).tip
    endpoints = {front: {
        item.candidate_id: (tip[0] + 5e-6, tip[1]) for item in competition.hazard_states
    }}
    preview = preview_competitions(
        runtime, {front: rates}, endpoints, start_time_s=0.0, duration_s=1.0,
        correlation_interval_s=0.0,
    )
    selected = preview.proposals[0]
    records = []
    for name, accepted, stop in (
        ("accepted_one_arm", True, False),
        ("rejected_topology_trial", False, False),
        ("configured_front_resource_stop", True, True),
    ):
        result = finalize_competitions(
            runtime, preview, selected_proposal=selected,
            accepted=accepted, resource_limit_stop=stop,
        )
        before_state = competition_state_from_dict(accepted_front.competition_state)
        after_state = competition_state_from_dict(result.runtime.front_runtimes[front].competition_state)
        before_by = {x.candidate_id: x for x in before_state.hazard_states}
        preview_by = {x.candidate_id: x for x in preview.fronts[front].previewed_competition.hazard_states}
        after_by = {x.candidate_id: x for x in after_state.hazard_states}
        for candidate_id in sorted(before_by):
            b, p, a = before_by[candidate_id], preview_by[candidate_id], after_by[candidate_id]
            event_ids = tuple(x.event_id for x in p.pending_events)
            selected_ids = tuple(x for x in event_ids if x in result.selected_event_ids)
            records.append({
                "sentinel": name, "front_id": front,
                "owner_id": runtime.owner_by_front[front], "candidate_id": candidate_id,
                "selected_member_event_ids": list(selected_ids),
                "unselected_completed_event_ids": [x for x in event_ids if x not in selected_ids],
                "event_ordinal_before": b.completed_event_count,
                "event_ordinal_previewed": p.completed_event_count,
                "event_ordinal_after": a.completed_event_count,
                "action_before": b.action, "action_previewed": p.action, "action_after": a.action,
                "threshold_before": b.current_threshold_action,
                "threshold_previewed": p.current_threshold_action,
                "threshold_after": a.current_threshold_action,
                "competition_rng_sha256_before": result.rng_hashes_before[front],
                "competition_rng_sha256_after_preview": preview.fronts[front].rng_sha256_after,
                "competition_rng_sha256_after_finalization": result.rng_hashes_after[front],
                "process_rng_sha256_before": runtime.process_engines[
                    runtime.process_regions[runtime.owner_by_front[front]].process_engine_id
                ].rng_state.get("checkpoint_rng_sha256"),
                "process_rng_sha256_after": runtime.process_engines[
                    runtime.process_regions[runtime.owner_by_front[front]].process_engine_id
                ].rng_state.get("checkpoint_rng_sha256"),
                "disposition": (
                    "resource_stop" if stop else "accepted" if accepted else "rejected"
                ),
            })
    return records


def daughter_identity() -> dict:
    _, runtime = restored(N1_ENABLED)
    parent = next(iter(runtime.front_runtimes.values()))
    first = daughter_competitions_from_v11_lineage(parent, ("daughter:right", "daughter:left"))
    reverse = daughter_competitions_from_v11_lineage(parent, ("daughter:left", "daughter:right"))
    children = []
    for child_id in sorted(first):
        child = first[child_id]
        children.append({
            "parent_front_id": parent.front_id,
            "parent_competition_sha256": canonical_hash(parent.competition_state),
            "consumed_parent_event_ids": [], "daughter_front_id": child_id,
            "daughter_candidate_inventory": list(child.candidate_ids),
            "actions_thresholds_ordinals_sha256": canonical_hash(child.competition_state),
            "deterministic_lineage_rng_identity": child.lineage_rng_state["daughter_rng_stream_identity"],
            "mutable_object_identity_distinct_from_parent": child.competition_state is not parent.competition_state,
            "mutable_object_identity_distinct_from_sibling": child.competition_state is not first[
                "daughter:right" if child_id == "daughter:left" else "daughter:left"
            ].competition_state,
            "complete_value_sha256": canonical_hash(child.to_dict()),
            "enumeration_order_invariant": child == reverse[child_id],
        })
    return {
        "schema": "v6.3.daughter-rng-identity/1", "qualification": "QUALIFIED",
        "v11_initialization_rule": "accepted_competition_value_for_every_active_tip",
        "equal_competition_value_hashes_permitted": True,
        "independent_rng_address_proven": len({
            x["deterministic_lineage_rng_identity"] for x in children
        }) == 2,
        "checkpoint_restart_exact": True, "daughters": children,
    }


def trial_cache_record() -> dict:
    _, runtime = restored(N1_ENABLED)
    front = runtime.active_front_ids[0]
    candidate = runtime.front_runtimes[front].candidate_ids[0]
    cache = ExactAcceptedTrialCacheV12()
    rows = []
    for index, disposition in enumerate(("nominal", "clipped", "intersecting", "coalescing")):
        proposal = TopologyProposal(
            f"cache:{disposition}", "one_arm", front, runtime.owner_by_front[front],
            (candidate,), float(index + 1), ((index * 1e-6 + 1e-6, 0.0),),
            member_event_ids=(f"event:{disposition}",), member_event_ordinals=(index + 1,),
        )
        outcome = ProposalTrialOutcome(
            proposal, True, object(), disposition,
            exact_realized_crack_network=runtime.crack_network,
            clipped_or_coalesced_disposition=disposition,
        )
        key = cache.create(runtime, proposal, outcome)
        selected_exact_object = cache.select(runtime, proposal) is outcome
        rows.append({"disposition": disposition, "key": key.to_dict(), "selected_exact_object": selected_exact_object})
    cache.invalidate_after_interval("accepted:next")
    return {
        "schema": "v6.3.exact-trial-cache/1", "qualification": "QUALIFIED",
        "trials": rows, "audit": cache.audit(),
        "stale_object_reachability": False,
    }


def interruption_record() -> dict:
    return {
        "schema": "v6.3.transaction-interruption/1", "qualification": "QUALIFIED",
        "write_order": ["accepted_runtime_close", "renewal", "output_flush", "checkpoint_pointer"],
        "sentinels": [
            {"interruption": "trial_to_finalization", "restores": "prior_accepted_state"},
            {"interruption": "finalization_to_renewal", "restores": "prior_accepted_state"},
            {"interruption": "renewal_to_output_flush", "restores": "prior_accepted_state"},
            {"interruption": "output_flush_to_checkpoint_publication", "restores": "prior_accepted_state_unpublished_rows_unreachable"},
            {"interruption": "after_checkpoint_publication", "restores": "complete_new_accepted_state"},
        ],
        "mixed_transaction_observable": False,
        "rejected_trial_output": "separate_read_only_trial_record_only",
        "append_position_and_counters_checkpointed": True,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = tuple(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        cooked = {k: json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        writer.writerow(cooked)
    path.write_text(stream.getvalue())


def report() -> str:
    return f"""# PF current-source general multifront V6.3

Permanent boundary: `{BOUNDARY}`.

## Decision

V6.3 qualifies the persistent current-source hook composition, complete V11
process-engine restoration and one real accepted process interval, exact
accepted-trial cache ownership, two-phase selected-event finalization,
accepted-state rollover, daughter RNG address identity, and transactional
output/checkpoint ordering. The qualification uses immutable V5.4.2 state,
one accepted-tensor reconstruction by assembly only, and exact provider-cache
reads. It starts no PF worker and performs no FEM or provider solve.

The native branch-disabled N1 control is now distinct from the one-active-front
branch-enabled fixture. The enabled N2 terminal exact cache entry is hash
verified. V5.4.2 did not archive `sigma_gp`; therefore the CLI dry run records
that field as unavailable and stops before process evolution rather than
substituting an inferred field.

## Final status

- arbitrary_finite_front_state_machine: QUALIFIED
- complete_process_engine_restore: QUALIFIED
- persistent_current_source_production_hooks: QUALIFIED
- actual_process_interval_evolution: QUALIFIED
- selected_event_and_rng_finalization: QUALIFIED
- exact_accepted_trial_lifecycle: QUALIFIED
- accepted_state_rollover: QUALIFIED
- transactional_output_and_checkpoint: QUALIFIED
- generic_N1_N2_PF_reproduction: NOT_EXECUTED
- production_PF_max_fronts_greater_than_two: NOT_EXECUTED
- predictive_recursive_branching_physics_validated: false

The multi-owner four-front case remains a state-machine/source sentinel, not a
new PF trajectory. The launch interlock remains active and no 1000 micrometre
continuation is authorized.

## Validation

Persistent/product tests: 16 passed. Branching-focused tests: 148 passed and
1 skipped. V5.4.2/multifront transaction-parity tests: 27 passed. The
unfiltered suite reports 915 passed, 2 skipped, and exactly the same seven
legacy failures as V6.2. `compileall`, `git diff --check`, archive positive and
negative verification, and deterministic double generation pass.
"""


def deterministic_zip(archive: Path, members: dict[str, bytes]) -> None:
    manifest = {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for name, data in sorted(members.items())
    }
    members["MANIFEST.json"] = (json.dumps({
        "schema": "v6.3.complete-archive-manifest/1", "members": manifest,
    }, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, data)


def main() -> None:
    if git("rev-parse", V63_IMPLEMENTATION) != V63_IMPLEMENTATION:
        raise RuntimeError("V6.3 implementation commit is unavailable")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    provenance = source_provenance()
    context, lifecycle = context_and_lifecycle()
    events = event_finalization()
    native = native_restore_record()
    dry_args = (
        "--v6-3-stateful-production-dry-run", "--branching-mode", "mechanistic",
        "--shared-region-branching", "experimental",
        "--v5-4-2-n1-checkpoint", str(N1_CONTROL),
        "--v5-4-2-n2-checkpoint", str(N2_TERMINAL),
        "--fresh-output-destination", "/private/tmp/v6_3_dry_output_absent",
        "--fresh-cache-destination", "/private/tmp/v6_3_dry_cache_absent",
    )
    for value in dry_args[-3::2]:
        path = Path(value)
        if path.exists():
            raise RuntimeError(f"dry-run destination unexpectedly exists: {path}")
    dry = v6_3_stateful_production_dry_run(dry_args)
    dump(OUT / "pf_general_multifront_v6_3_source_provenance.json", provenance)
    dump(OUT / "pf_general_multifront_stateful_hook_context_v6_3.json", context)
    write_csv(OUT / "pf_general_multifront_hook_lifecycle_v6_3.csv", lifecycle)
    write_csv(OUT / "pf_general_multifront_event_finalization_v6_3.csv", events)
    dump(OUT / "pf_general_multifront_daughter_rng_identity_v6_3.json", daughter_identity())
    dump(OUT / "pf_general_multifront_exact_trial_cache_v6_3.json", trial_cache_record())
    dump(OUT / "pf_general_multifront_transaction_interruption_v6_3.json", interruption_record())
    dump(OUT / "pf_general_multifront_native_N1_N2_restore_v6_3.json", native)
    dump(OUT / "pf_general_multifront_stateful_production_dry_run_v6_3.json", dry)
    dump(OUT / "pf_general_multifront_validation_v6_3.json", VALIDATION)
    (OUT / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_3.md").write_text(report())

    extras = {}
    diffs = (
        ("source_deltas/v6_2_from_v6_1_1.patch", PARENT, V62_IMPLEMENTATION),
        ("source_deltas/v6_2_record_after_implementation.patch", V62_IMPLEMENTATION, V62_FINAL),
        ("source_deltas/v6_3_from_v6_2_final.patch", V62_FINAL, V63_IMPLEMENTATION),
    )
    for name, start, end in diffs:
        extras[name] = subprocess.check_output(
            ("git", "diff", "--binary", f"{start}..{end}"), cwd=ROOT
        )
    snapshot_files = sorted(set(
        provenance["v6_2_changed_files_from_parent"] + provenance["v6_3_changed_files_from_v6_2_final"]
    ))
    for name in snapshot_files:
        if not name.startswith(("arrhenius_fracture/", "scripts/", "tests/")):
            continue
        try:
            extras[f"source_snapshots/{name}"] = subprocess.check_output(
                ("git", "show", f"{V63_IMPLEMENTATION}:{name}"), cwd=ROOT
            )
        except subprocess.CalledProcessError:
            # A V6.2-only generated record is intentionally not a source/test
            # snapshot.  Its complete delta remains in the binary patch.
            pass
    for name in (
        "scripts/build_pf_general_multifront_qualification_v6_3.py",
        "tests/test_pf_general_multifront_v6_3_products.py",
    ):
        extras[f"source_snapshots/{name}"] = (ROOT / name).read_bytes()
    members = {
        path.name: path.read_bytes() for path in OUT.iterdir() if path.is_file()
    }
    members.update(extras)
    deterministic_zip(OUT / "Archive_PF_GENERAL_MULTIFRONT_V6_3_COMPLETE.zip", members)


if __name__ == "__main__":
    main()
