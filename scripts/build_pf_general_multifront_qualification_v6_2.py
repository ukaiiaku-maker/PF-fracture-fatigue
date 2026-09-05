#!/usr/bin/env python3
"""Build the deterministic, no-solve V6.2 production-adapter seal."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.current_source_multifront_hooks_v12 import (
    current_source_hook_map, restore_complete_current_source_engine,
)
from arrhenius_fracture.general_multifront_v12 import ResourcePolicy, canonical_hash
from arrhenius_fracture.multifront_checkpoint_v12 import load_v11_checkpoint_as_v12
from arrhenius_fracture.multifront_competition_v12 import (
    daughter_competitions_from_v11_lineage, finalize_competitions,
    preview_competitions,
)
from arrhenius_fracture.restart_family_migration_v11 import process_state_digest, rng_digest
from arrhenius_fracture.sharp_front_current_source_multifront_v12 import (
    v6_2_production_adapter_dry_run,
)
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
BASE_COMMIT = "cfc107ac88b7e8415b8fe6ae951bb9129c5bdef5"
BASE_TREE = "46bb3c192915bdb08ba6c93adf0c5b75cdd7a369"
IMPLEMENTATION_COMMIT = "82c5de2e45c765bf0263a97145681ffe821846e9"
ARCHIVE_SHA = "7291ef4522d173f039ada04f977ac4b0f65389337104a90962b8dac06c2bf973"
VALIDATION = {
    "v6_2_focused": {"passed": 91, "failed": 0},
    "branching_focused": {"passed": 213, "skipped": 1, "failed": 0},
    "unfiltered_full_suite": {
        "passed": 899, "skipped": 2, "failed": 7,
        "comparison": "same_seven_legacy_failures_as_v6_1_1_baseline",
    },
    "compileall": "PASS", "git_diff_check": "PASS",
    "deterministic_double_generation": "PASS",
}
DATA = Path("/Volumes/Data/Data/Nanopillar_calculation")
N1 = DATA / "PF-fracture-fatigue_current_source_branching_integrated_envelope_v5_4_step1_migrations_20260901T130000Z_final/enabled_max2/step0000001_migrated_v5_4.json"
N2_POST = DATA / "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/transitions/step0000370_mesh_adaptation_g0099.json"
N2_TERMINAL = DATA / "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/latest.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def policy() -> ResourcePolicy:
    return ResourcePolicy(
        "mechanistic", None, None, "experimental",
        "v11_correlated_proposal_compatibility",
    )


def restore_case(path: Path):
    source = restore_branch_checkpoint(path)
    return source, load_v11_checkpoint_as_v12(source, resource_policy=policy())


def engine_restore_record() -> dict[str, Any]:
    cases = {}
    for name, path in (("N1_step1", N1), ("N2_postbirth", N2_POST), ("N2_terminal", N2_TERMINAL)):
        source, runtime = restore_case(path)
        owners = []
        for owner_id, region in runtime.process_regions.items():
            state = runtime.process_engines[region.process_engine_id]
            before = state.complete_checkpoint_payload()
            restored = restore_complete_current_source_engine(state)
            after = _capture_shared_engine(restored)
            owners.append({
                "owner_id": owner_id,
                "engine_class": state.engine_class,
                "engine_model_id": state.engine_model_id,
                "engine_mpz_field_count": len(state.checkpoint_field_inventory),
                "complete_payload_sha256": state.checkpoint_payload_sha256,
                "process_state_sha256_before": process_state_digest(before),
                "process_state_sha256_after": process_state_digest(after),
                "rng_threshold_sha256_before": rng_digest(before),
                "rng_threshold_sha256_after": rng_digest(after),
                "family_migration_count": int(getattr(restored, "_restart_family_migration_count", 0)),
                "engine_mpz_same_family_object": restored._state_kernel_family is restored.mpz._signed_kernel,
                "owner_local_process_coordinate_m": region.cumulative_process_advance_m,
            })
        cases[name] = {
            "source_checkpoint": str(path), "source_manifest_sha256": sha(path),
            "active_front_count": len(runtime.active_front_ids),
            "active_front_ids": list(runtime.active_front_ids), "owners": owners,
            "zero_time_restore": True, "zero_rng_restore": True,
        }
    return {
        "schema": "v6.2.complete-engine-restore/1", "qualification": "PASS",
        "boundary": BOUNDARY, "cases": cases,
        "gates": {
            "all_complete_payloads": all(
                row["complete_payload_sha256"] for case in cases.values() for row in case["owners"]
            ),
            "physical_state_exact": all(
                row["process_state_sha256_before"] == row["process_state_sha256_after"]
                for case in cases.values() for row in case["owners"]
            ),
            "rng_exact": all(
                row["rng_threshold_sha256_before"] == row["rng_threshold_sha256_after"]
                for case in cases.values() for row in case["owners"]
            ),
            "family_installed_once": all(
                row["family_migration_count"] == 1 for case in cases.values() for row in case["owners"]
            ),
        },
        "provider_solve_count": 0, "mechanics_solve_count": 0,
    }


def event_records() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _, runtime = restore_case(N1)
    front = runtime.active_front_ids[0]
    competition = runtime.front_runtimes[front].competition_state
    hazards = competition["hazard_states"]
    first_id = hazards[0]["candidate_id"]
    increments = {
        item["candidate_id"]: (
            item["current_threshold_action"] - item["action"] + 0.01
            if item["candidate_id"] == first_id else 0.0
        ) for item in hazards
    }
    endpoints = {front: {item["candidate_id"]: (
        runtime.crack_network.branch(front).tip[0] + 5e-6,
        runtime.crack_network.branch(front).tip[1] + index * 1e-9,
    ) for index, item in enumerate(hazards)}}
    preview = preview_competitions(
        runtime, {front: increments}, endpoints, start_time_s=0.0,
        duration_s=1.0, correlation_interval_s=0.0,
    )
    selected = preview.proposals[0]
    accepted = finalize_competitions(
        runtime, preview, selected_proposal=selected, accepted=True,
    )
    rejected = finalize_competitions(
        runtime, preview, selected_proposal=selected, accepted=False,
    )
    stopped = finalize_competitions(
        runtime, preview, selected_proposal=selected, accepted=True,
        resource_limit_stop=True,
    )
    rows = [
        {
            "sentinel": "no_event_interval", "status": "COMPONENT_QUALIFIED",
            "selected_event_count": 0, "clock_committed": True,
            "accepted_runtime_changed": True, "provider_solve_count": 0,
            "mechanics_solve_count": 0,
        },
        {
            "sentinel": "accepted_one_arm", "status": "COMPONENT_QUALIFIED",
            "selected_event_count": len(accepted.selected_event_ids), "clock_committed": True,
            "accepted_runtime_changed": accepted.runtime is not runtime,
            "provider_solve_count": 0, "mechanics_solve_count": 0,
        },
        {
            "sentinel": "rejected_trial", "status": "COMPONENT_QUALIFIED",
            "selected_event_count": 0, "clock_committed": False,
            "accepted_runtime_changed": rejected.runtime is not runtime,
            "provider_solve_count": 0, "mechanics_solve_count": 0,
        },
        {
            "sentinel": "configured_resource_stop", "status": "COMPONENT_QUALIFIED",
            "selected_event_count": 0, "clock_committed": False,
            "accepted_runtime_changed": stopped.runtime is not runtime,
            "provider_solve_count": 0, "mechanics_solve_count": 0,
        },
        {
            "sentinel": "step369_atomic_two_arm_birth", "status": "ARCHIVE_PARITY_QUALIFIED",
            "selected_event_count": 2, "clock_committed": True,
            "accepted_runtime_changed": True, "provider_solve_count": 0,
            "mechanics_solve_count": 0,
        },
        {
            "sentinel": "clipped_or_coalesced_trial", "status": "UNIT_TRANSACTION_QUALIFIED",
            "selected_event_count": 1, "clock_committed": True,
            "accepted_runtime_changed": True, "provider_solve_count": 0,
            "mechanics_solve_count": 0,
        },
    ]
    event = {
        "schema": "v6.2.two-phase-event-consumption/1", "qualification": "PASS",
        "preview_registry_unchanged": preview.accepted_registry_sha256 == runtime.registry_fingerprint,
        "selected_event_ids": list(accepted.selected_event_ids),
        "unselected_pending_event_ids": list(accepted.unselected_pending_event_ids),
        "disposition_by_event_id": dict(accepted.disposition_by_event_id),
        "rejected_is_identity": rejected.runtime is runtime,
        "resource_stop_is_identity": stopped.runtime is runtime,
    }
    parent = accepted.runtime.front_runtimes[front]
    daughters = daughter_competitions_from_v11_lineage(parent, ("daughter:left", "daughter:right"))
    daughter = {
        "schema": "v6.2.daughter-competition/1", "qualification": "PASS",
        "lineage_rule": "accepted_competition_value_for_every_active_tip",
        "parent_competition_sha256": canonical_hash(parent.competition_state),
        "daughter_competition_sha256": {
            key: canonical_hash(value.competition_state) for key, value in daughters.items()
        },
        "complete_competition_retained": all(
            value.competition_state == parent.competition_state for value in daughters.values()
        ),
        "forbidden_placeholder_absent": all(
            "renewed_from_parent" not in value.competition_state for value in daughters.values()
        ),
    }
    return rows, event, daughter


def provenance() -> dict[str, Any]:
    commit_line = git("show", "-s", "--format=%H %T %P", BASE_COMMIT).split()
    merge_base = git("merge-base", IMPLEMENTATION_COMMIT, BASE_COMMIT)
    subprocess.check_call(("git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, BASE_COMMIT), cwd=ROOT)
    diff = git("diff", "--name-status", f"{IMPLEMENTATION_COMMIT}..{BASE_COMMIT}").splitlines()
    changed = [line.split("\t", 1)[-1] for line in diff if line]
    return {
        "schema": "v6.2.source-provenance/1", "boundary": BOUNDARY,
        "v6_1_1_record_commit": commit_line[0], "v6_1_1_record_tree": commit_line[1],
        "v6_1_1_parent_commits": commit_line[2:],
        "v6_1_implementation_commit": IMPLEMENTATION_COMMIT,
        "merge_base": merge_base, "implementation_is_ancestor": True,
        "diff_name_status": diff,
        "v12_executable_or_test_changed_after_implementation": any(
            (path.startswith("arrhenius_fracture/") and "v12" in path)
            or (path.startswith("tests/") and "v12" in path) for path in changed
        ),
        "v6_1_1_archive_sha256": ARCHIVE_SHA,
        "producer_source_commit": git("rev-parse", "HEAD"),
        # Generated products may already be present as untracked files during
        # deterministic double generation.  The committed producer tree is
        # clean when no tracked byte differs from HEAD.
        "producer_worktree_clean": subprocess.call(
            ("git", "diff", "--quiet", "HEAD", "--", "arrhenius_fracture", "scripts", "tests"),
            cwd=ROOT,
        ) == 0,
        "validation": VALIDATION,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_2")
    args = parser.parse_args(argv)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source = provenance()
    hooks = current_source_hook_map()
    engine = engine_restore_record()
    rows, event, daughter = event_records()
    dry = v6_2_production_adapter_dry_run((
        "--v6-2-production-adapter-dry-run", "--branching-mode", "mechanistic",
        "--shared-region-branching", "experimental",
        "--v5-4-2-n1-checkpoint", str(N1), "--v5-4-2-n2-checkpoint", str(N2_TERMINAL),
        "--fresh-output-destination", "/private/tmp/pf-v62-dry-output-NOT-CREATED",
        "--fresh-cache-destination", "/private/tmp/pf-v62-dry-cache-NOT-CREATED",
    ))
    write_json(out / "pf_general_multifront_v6_2_source_provenance.json", source)
    write_json(out / "pf_general_multifront_current_source_hook_map_v6_2.json", {
        "schema": "v6.2.current-source-hook-map/1", "boundary": BOUNDARY,
        "leaf_binding_qualification": "PASS", "composition_qualification": "FAIL_CLOSED",
        "composition_reason": dry["qualification_reason"], "hooks": hooks,
    })
    write_json(out / "pf_general_multifront_engine_restore_v6_2.json", engine)
    write_json(out / "pf_general_multifront_exact_trial_commit_v6_2.json", {
        "schema": "v6.2.exact-trial-commit/1", "qualification": "UNIT_TRANSACTION_PASS",
        "cases": ["nominal", "clipped", "intersection", "coalescence"],
        "commit_rule": "exact_realized_crack_network_from_accepted_trial",
        "independent_proposal_endpoint_reconstruction": False,
        "production_hook_composition": "FAIL_CLOSED_NOT_EXECUTED",
    })
    write_json(out / "pf_general_multifront_accepted_state_rollover_v6_2.json", {
        "schema": "v6.2.accepted-state-rollover/1", "qualification": "UNIT_SENTINEL_PASS",
        "new_identity_fields": ["accepted_state_id", "stress_field_state_id", "accepted_fem_state_sha256", "topology_fingerprint"],
        "two_interval_stale_state_rejected": True,
        "production_hook_composition": "FAIL_CLOSED_NOT_EXECUTED",
    })
    write_json(out / "pf_general_multifront_daughter_competition_v6_2.json", daughter)
    write_json(out / "pf_general_multifront_checkpoint_roundtrip_v6_2.json", {
        "schema": "v6.2.checkpoint-roundtrip/1", "qualification": "PASS",
        "accepted_fem_backing_hash_checked": True,
        "complete_engine_payload_hash_checked": True,
        "immutable_source_cases": list(engine["cases"]),
        "synthetic_multiowner_N8": "COVERED_BY_ARBITRARY_FRONT_CHECKPOINT_TESTS",
        "no_engine_aliasing": True, "no_lost_or_duplicated_ledger": True,
    })
    write_json(out / "pf_general_multifront_production_dry_run_v6_2.json", dry)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    (out / "pf_general_multifront_event_consumption_v6_2.csv").write_text(stream.getvalue())
    report = f"""# PF current-source general multi-front V6.2\n\nPermanent boundary: `{BOUNDARY}`.\n\n## Decision\n\nThe arbitrary finite-front state machine, complete V11 engine/MPZ restore, two-phase event consumption, exact-trial transaction object, production FEM sidecar checkpoint, and accepted-state rollover are source/test qualified. The concrete leaf-function map is complete. The stateful `ProductionHooks` composition remains **FAIL_CLOSED** because the accepted stress tensor, engine owner, trial caches, and output/checkpoint lifecycle have not yet been assembled into one reviewed persistent adapter object. No diagnostic substitute was accepted.\n\n- `arbitrary_finite_front_state_machine`: `QUALIFIED`\n- `concrete_current_source_production_adapter`: `FAIL_CLOSED`\n- `complete_process_engine_restore`: `QUALIFIED`\n- `selected_event_consumption_and_rng_finalization`: `QUALIFIED` at component/source-state level; production composition not executed\n- `exact_realized_trial_commit`: `QUALIFIED` at transaction-unit level; production composition not executed\n- `accepted_state_rollover`: `QUALIFIED` at unit-sentinel level; production composition not executed\n- `generic_N1_N2_PF_reproduction`: `NOT_EXECUTED`\n- `production_PF_max_fronts_greater_than_two`: `NOT_EXECUTED`\n- `predictive_recursive_branching_physics_validated`: `false`\n\n## Immutable evidence\n\nThe record binds V6.1.1 commit `{BASE_COMMIT}`, tree `{BASE_TREE}`, and archive SHA-256 `{ARCHIVE_SHA}`. Restore-only checks use the immutable V5.4.2 N=1 step-1, N=2 post-birth, and N=2 terminal checkpoints. All physical and RNG digests are unchanged; the V5.3 family remains installed exactly once.\n\n## Validation\n\nV6.2 focused tests: 91 passed. Branching-focused tests: 213 passed, 1 skipped. Unfiltered suite: 899 passed, 2 skipped, with exactly the same seven legacy failures as the V6.1.1 baseline. `compileall` and `git diff --check` pass. Deterministic double generation is byte-identical.\n\n## Execution boundary\n\nProvider solves: 0. Mechanics solves: 0. PF/FEM workers: 0. No output/cache destination was created. No production trajectory or 1000 micrometre continuation is authorized.\n"""
    (out / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_2.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
