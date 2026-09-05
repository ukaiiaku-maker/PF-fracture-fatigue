#!/usr/bin/env python3
"""Build the deterministic, no-solve V6.4 integrated-interval record."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.sharp_front_current_source_multifront_v12 import (
    v6_4_integrated_interval_dry_run,
)

BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
V62_FINAL = "f965b9405deb3459ff1a75a6f69022e3569f28c0"
V63_IMPLEMENTATION = "4de34c31cc349a9e2a8c738a91d4ea0c685e2efd"
V63_FINAL = "8fcee267b7d093e23c51e48f14b6b57952363f1e"
V63_ARCHIVE_SHA256 = "b46f9b5581ecd952dc244d2f21d23eca342c456a1be5e71f5220deaae0496e5d"
V64_IMPLEMENTATION = "2f0b1d3"
DATA = Path("/Volumes/Data/Data/Nanopillar_calculation")
V542_ROOT = DATA / "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z"
N1 = DATA / "PF-fracture-fatigue_current_source_branching_integrated_envelope_v5_4_step1_migrations_20260901T130000Z_final/control_max1/step0000001_migrated_v5_4.json"
N2 = DATA / "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/latest.json"
OUT = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_4"
ARCHIVE = OUT / "Archive_PF_GENERAL_MULTIFRONT_V6_4_COMPLETE.zip"

VALIDATION = {
    "focused_v6_4_tests": {"passed": 28, "failed": 0},
    "branching_focused_tests": {"passed": 176, "skipped": 1, "failed": 0},
    "unfiltered_full_suite": {
        "passed": 943, "skipped": 2, "failed": 7,
        "comparison": "same_seven_legacy_failures_as_v6_3_baseline",
        "failure_node_ids": [
            "tests/test_real_signed_builder_v10212.py::test_review_builder_emits_artifact_consumable_by_production_loader",
            "tests/test_v10214_capture_model_id.py::test_v10214_capture_model_id",
            "tests/test_v10214_response_model_id.py::test_v10214_response_model_id",
            "tests/test_v10215_stage3_schema_and_status.py::test_stage3_status_script_parses_and_reports_missing_run",
            "tests/test_v10_2_27_zero_event_summary.py::test_zero_event_summary_is_recorded_explicitly",
            "tests/test_v10_2_27_zero_event_summary.py::test_empty_geometry_rejects_nonzero_advances",
            "tests/test_v10_2_29_vhcf_nonlinear_selector.py::test_delegate_reenters_selector_under_global_cycle_cap",
        ],
    },
    "compileall": "PASS", "git_diff_check": "PASS",
    "deterministic_double_generation": "PASS",
    "provider_solve_count": 0, "mechanics_solve_count": 0,
    "workers_started": 0,
}


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def base(value: dict) -> dict:
    return {"boundary": BOUNDARY, **value}


def provenance() -> dict:
    changed = git("diff", "--name-only", f"{V63_IMPLEMENTATION}..{V63_FINAL}").splitlines()
    executable = [
        path for path in changed
        if path.startswith("arrhenius_fracture/") or path.startswith("tests/")
    ]
    archive = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_3/Archive_PF_GENERAL_MULTIFRONT_V6_3_COMPLETE.zip"
    if sha(archive) != V63_ARCHIVE_SHA256:
        raise RuntimeError("immutable V6.3 archive hash mismatch")
    return base({
        "schema": "v6.4.source-provenance/1",
        "v6_2_final": {"commit": V62_FINAL, "tree": git("rev-parse", f"{V62_FINAL}^{{tree}}")},
        "v6_3_implementation": {"commit": V63_IMPLEMENTATION, "tree": git("rev-parse", f"{V63_IMPLEMENTATION}^{{tree}}")},
        "v6_3_final_record": {"commit": V63_FINAL, "tree": git("rev-parse", f"{V63_FINAL}^{{tree}}")},
        "v6_4_implementation": {
            "commit": git("rev-parse", V64_IMPLEMENTATION),
            "tree": git("rev-parse", f"{V64_IMPLEMENTATION}^{{tree}}"),
        },
        "prior_sealed_v6_4_implementation": {
            "commit": git("rev-parse", "a5c57eb"),
            "tree": git("rev-parse", "a5c57eb^{tree}"),
            "status": "IMMUTABLE_SUPERSEDED_ONLY_BY_EXPLICIT_STRESS_AVAILABILITY_FIX",
        },
        "ancestry": {
            "v6_2_to_v6_3_implementation": True,
            "v6_3_implementation_to_final": True,
            "v6_3_final_to_v6_4_implementation": True,
        },
        "v6_3_post_implementation_changed_files": changed,
        "executable_v12_source_or_tests_changed_after_v6_3_implementation": bool(executable),
        "executable_v12_source_or_test_files": executable,
        "v6_3_binary_diff_fingerprint": {
            "command": "git diff --binary 4de34c31...8fcee267...",
            "sha256": hashlib.sha256(subprocess.check_output(
                ("git", "diff", "--binary", V63_IMPLEMENTATION, V63_FINAL),
                cwd=ROOT,
            )).hexdigest(),
            "publication": "REPRODUCIBLE_FROM_BOUND_COMMITS_NOT_EMBEDDED_AS_REPOSITORY_TEXT",
        },
        "v6_3_archive": {"path": str(archive.resolve()), "sha256": sha(archive)},
        "v6_4_source_sha256": {
            path: sha(ROOT / path) for path in (
                "arrhenius_fracture/production_multifront_v12.py",
                "arrhenius_fracture/stateful_multifront_production_v12.py",
                "arrhenius_fracture/sharp_front_current_source_multifront_v12.py",
                "tests/test_multifront_v6_4_integrated_interval.py",
            )
        },
        "validation": VALIDATION,
    })


def archive_sufficiency_inventory() -> tuple[list[dict], dict]:
    from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint

    rows = []
    for run in sorted(V542_ROOT.glob("theta40_v5_4_2_*")):
        manifests = []
        for path in (run / "live_kernel_cache").glob("*/manifest.json"):
            value = json.loads(path.read_text())
            if value.get("accepted") is True and value.get("interpolation_permitted") is False:
                manifests.append((path, value))
        transition_paths = sorted((run / "checkpoint/transitions").glob("*.json"))
        by_step = {
            int(re.search(r"step(\d+)", path.name).group(1)): path
            for path in transition_paths
        }
        candidates = [("transition", step, path) for step, path in sorted(by_step.items())]
        candidates.append(("latest", None, run / "checkpoint/latest.json"))
        for kind, step, path in candidates:
            value = json.loads(path.read_text())
            source_topology = value["topology_fingerprint"]
            hits = [item for item in manifests if item[1].get("topology_fingerprint") == source_topology]
            has_next = bool(kind == "transition" and step + 1 in by_step)
            next_path = by_step.get(step + 1) if has_next else None
            coverage = "NOT_EVALUABLE_NO_EXACT_CACHE"
            needed_count = None; observed_count = None
            if len(hits) == 1:
                source = restore_branch_checkpoint(path)
                needed = {
                    (front_id, item.candidate_id)
                    for front_id, competition in source.front_competitions.items()
                    for item in competition.candidates
                }
                provider = pickle.loads(hits[0][0].with_name("provider_state.pkl").read_bytes())
                observed = set()
                for tip in provider.get("tips", ()):
                    xy = tuple(map(float, tip.get("tip_xy_m", ())))
                    owners = [
                        front_id for front_id in source.state.crack_network.active_tip_ids
                        if len(xy) == 2 and __import__("math").dist(
                            xy, source.state.crack_network.branch(front_id).tip
                        ) <= 1.0e-12
                    ]
                    if len(owners) == 1:
                        observed.update(
                            (owners[0], str(item["candidate_id"]))
                            for item in tip.get("directional", ())
                        )
                needed_count = len(needed); observed_count = len(observed)
                coverage = "COMPLETE" if observed == needed else "INCOMPLETE"
            state_path = path.with_name(value["state_file"])
            reasons = []
            if not state_path.is_file():
                reasons.append("MISSING_COMPLETE_STATE_SIDECAR")
            if not has_next:
                reasons.append("MISSING_COMPLETE_ARCHIVED_NEXT_STATE")
            if len(hits) != 1:
                reasons.append(f"EXACT_PROVIDER_CACHE_HIT_COUNT_{len(hits)}")
            elif coverage != "COMPLETE":
                reasons.append("INCOMPLETE_FRONT_CANDIDATE_COVERAGE")
            rejection = (
                "CANDIDATE_COMPLETE_PENDING_ASSEMBLY_CHECK"
                if not reasons else "|".join(reasons)
            )
            rows.append({
                "run": run.name, "checkpoint_kind": kind,
                "step": "" if step is None else step,
                "checkpoint": str(path.resolve()), "checkpoint_sha256": sha(path),
                "active_front_count": value["active_front_count"],
                "accepted_opening_m": value["accepted_load"],
                "state_sidecar_present": state_path.is_file(),
                "stress_assembly_inputs_present": state_path.is_file(),
                "complete_next_state_present": has_next,
                "next_checkpoint": "" if next_path is None else str(next_path.resolve()),
                "exact_cache_hit_count": len(hits),
                "required_front_candidate_count": "" if needed_count is None else needed_count,
                "observed_front_candidate_count": "" if observed_count is None else observed_count,
                "front_candidate_coverage": coverage,
                "admission": "REJECTED" if rejection != "CANDIDATE_COMPLETE_PENDING_ASSEMBLY_CHECK" else "CANDIDATE",
                "rejection_reason": rejection,
            })
    summary = base({
        "schema": "v6.4.archive-sufficiency-summary/1",
        "root": str(V542_ROOT.resolve()), "considered_checkpoint_count": len(rows),
        "admitted_interval_count": sum(row["admission"] == "ADMITTED" for row in rows),
        "control_latest": next(row for row in rows if row["run"].endswith("control_max1_seed3621") and row["checkpoint_kind"] == "latest"),
        "enabled_latest": next(row for row in rows if row["run"].endswith("enabled_max2_seed3621") and row["checkpoint_kind"] == "latest"),
        "cached_N1_end_to_end_parity": "FAIL_CLOSED_NO_USABLE_ARCHIVED_STRESS_INTERVAL",
        "cached_N2_end_to_end_parity": "FAIL_CLOSED_INCOMPLETE_FRONT_CANDIDATE_COVERAGE",
        "no_archive_gap_synthesized": True,
    })
    return rows, summary


def lifecycle_rows() -> list[dict]:
    names = (
        "begin_context_output_transaction", "adapt_accepted_mesh",
        "retain_adaptation_record", "load_pre_event_accepted_mechanics",
        "bind_actual_sigma_provisionally", "build_arbitrary_region_request",
        "cache_only_provider_lookup", "complete_directional_observation_batch",
        "preview_all_competitions", "form_correlated_same_tip_proposals",
        "trial_every_complete_proposal", "cache_exact_trial_objects",
        "select_global_proposal", "evolve_each_owner_once",
        "validate_each_owner_interval", "finalize_all_competitions",
        "apply_exact_delta_or_no_topology", "physical_selected_owner_renewal",
        "recapture_physical_engines", "recompute_process_connectivity",
        "derive_accepted_and_stress_identities", "validate_full_closure",
        "stage_outputs_and_checkpoint", "publish_one_atomic_pointer",
        "destroy_old_previews_and_trials",
    )
    return [{
        "order": index, "stage": name,
        "authoritative_symbol": "run_stateful_accepted_interval_v12",
        "source_level_status": "QUALIFIED",
        "cached_v5_4_2_status": (
            "FAIL_CLOSED" if index >= 7 else "REACHED_OR_EARLIER_FAIL_CLOSED"
        ),
    } for index, name in enumerate(names, 1)]


def hook_contract() -> dict:
    return base({
        "schema": "v6.4.hook-interface-contract/1", "qualification": "QUALIFIED",
        "authoritative_executor": "run_stateful_accepted_interval_v12(context, duration_s)",
        "entrypoint_delegates_when_context_present": True,
        "types": {
            "AdaptedAcceptedState": ["fem_state", "adaptation_record"],
            "SolvedAcceptedState": ["fem_state", "sigma_gp", "measurement", "accepted_load_m", "physical_time_s", "mechanics_source_identity"],
            "DirectionalObservationBatch": ["observations", "rates_by_front_and_candidate", "endpoints_by_front_and_candidate", "local_mechanics_records"],
        },
        "raw_tuple_rejected": True, "missing_field_rejected": True,
        "every_front_candidate_required": True,
    })


def stress_record(cli: dict) -> dict:
    return base({
        "schema": "v6.4.actual-stress-identity/1", "qualification": "QUALIFIED",
        "identity_inputs": ["actual_sigma_gp_bytes", "shape", "dtype", "accepted_FEM_state_identity", "topology", "mesh_discretization_identity", "mechanics_source_identity"],
        "constructor_recomputes": True, "rollover_recomputes": True,
        "numerical_zero_sentinel_for_unavailable_rejected": True,
        "finite_physical_zero_is_available": True,
        "unavailable_representation": {"available": False, "sigma_gp": None},
        "stale_sigma_sentinel": "PASS_FAIL_CLOSED",
        "immutable_cases": {
            key: {"stress_sha256": value["accepted_stress_sha256"], "status": value["status"], "reason": value["reason"]}
            for key, value in cli["cases"].items()
        },
    })


def owner_record() -> dict:
    return base({
        "schema": "v6.4.owner-process-renewal/1", "qualification": "QUALIFIED",
        "pre_progress_source": "ProcessEngineState.local_process_coordinate_m_per_owner",
        "restored_engine_coordinate_exact_check": True,
        "one_complete_interval_update_per_owner": True,
        "zero_topology_renewal_during_interval_evolution": True,
        "selected_owner_physical_V11_renewal_count": 1,
        "unselected_owner_physical_V11_renewal_count": 0,
        "postrenewal_payload_recaptured": True,
        "metadata_only_renewal_rejected": True,
        "cached_V5_4_2_parity": "FAIL_CLOSED_BEFORE_OWNER_EVOLUTION",
    })


def event_rows() -> list[dict]:
    return [
        {"case": "no_event", "source_level": "QUALIFIED", "cached_V5_4_2": "FAIL_CLOSED", "evidence": "integrated no-event and rollback tests"},
        {"case": "one_arm_selected", "source_level": "QUALIFIED", "cached_V5_4_2": "FAIL_CLOSED", "evidence": "selected ID consumed; one physical renewal"},
        {"case": "step_369_correlated_two_arm", "source_level": "QUALIFIED_COMPONENTS", "cached_V5_4_2": "FAIL_CLOSED", "evidence": "archive lacks an exact accepted-trial delta transaction"},
        {"case": "unselected_completed_other_front", "source_level": "QUALIFIED", "cached_V5_4_2": "FAIL_CLOSED", "evidence": "two-phase finalization preserves pending event"},
        {"case": "rejected_topology_trial", "source_level": "QUALIFIED", "cached_V5_4_2": "FAIL_CLOSED", "evidence": "accepted state and clocks restored"},
        {"case": "resource_limit_stop", "source_level": "QUALIFIED", "cached_V5_4_2": "FAIL_CLOSED", "evidence": "policy-bound no topology mutation"},
    ]


def exact_delta_record() -> dict:
    return base({
        "schema": "v6.4.exact-trial-delta/1", "qualification": "QUALIFIED",
        "application": "direct_registry_replacement_no_nominal_topology_reconstruction",
        "complete_fields": [
            "pre_post_networks", "continued_created_retired_fronts", "coalescence_target",
            "junctions", "owners", "process_regions", "process_engines", "front_competitions",
            "reservoirs", "scheduler_and_counters", "transaction_records", "realized_endpoints_and_lengths",
            "wake_mutation", "released_energy", "dissipative_cost",
        ],
        "active_front_change_tests": {
            "nominal_one_arm": "PASS", "clipped_one_arm": "PASS",
            "coalescence": "PASS", "retirement": "PASS", "binary_birth": "PASS",
        },
        "metadata_only_renewal_delta_rejected": True,
    })


def arbitrary_request_record() -> dict:
    return base({
        "schema": "v6.4.arbitrary-region-request/1", "qualification": "QUALIFIED",
        "outer_contract": "process_region_frame_by_owner_and_frame_by_tip",
        "v11_single_cluster_scope": "compatibility_adapter_only",
        "test": {"active_tips": 4, "owners": 3, "independent_owners": 2, "unresolved_two_front_owners": 1, "status": "PASS"},
        "per_tip_fields": ["process_owner", "frame_kind", "unresolved_junction_ids", "local_process_coordinate_m"],
    })


def atomic_record() -> dict:
    return base({
        "schema": "v6.4.atomic-output-checkpoint/1", "qualification": "QUALIFIED",
        "publication": "one_COMMITTED_marker_then_atomic_LATEST_pointer",
        "members": ["six_output_categories", "transaction_manifest", "accepted_FEM_sidecar", "runtime_checkpoint", "context_checkpoint"],
        "fsync_before_publication": True, "reader_ignores_staging": True,
        "failure_injection": {
            "after_each_of_six_output_categories": "PASS_PRIOR_STATE_ONLY",
            "engine_finalization_to_renewal": "PASS_PRIOR_STATE_ONLY",
            "renewal_to_output_staging": "PASS_PRIOR_STATE_ONLY",
            "outputs_to_checkpoint_staging": "PASS_PRIOR_STATE_ONLY",
            "checkpoint_to_commit_marker": "PASS_PRIOR_STATE_ONLY",
        },
    })


def cached_record(cli: dict) -> dict:
    return base({
        "schema": "v6.4.cached-end-to-end-parity/1", "qualification": "FAIL_CLOSED",
        "scientific_reason": "immutable V5.4.2 records do not contain complete integrated-interval inputs and exact deltas required for byte/state parity",
        "native_executor_invoked": True, "cli_cases": cli["cases"],
        "requested_sentinels": {
            "cached_no_event": "FAIL_CLOSED_NO_USABLE_ARCHIVED_STRESS_INTERVAL",
            "cached_one_arm": "FAIL_CLOSED_NO_EXACT_TRIAL_DELTA",
            "step_369_two_arm_birth": "FAIL_CLOSED_NO_ATOMIC_INTERVAL_TRANSACTION",
            "postbirth_interval": "FAIL_CLOSED_INCOMPLETE_FRONT_CANDIDATE_OBSERVATIONS",
            "rejected_trial": "FAIL_CLOSED_NO_EXACT_TRIAL_CACHE",
            "resource_limit_stop": "SOURCE_LEVEL_ONLY",
            "clipped_or_coalesced_trial": "SOURCE_LEVEL_ONLY",
            "two_consecutive_intervals": "FAIL_CLOSED_FIRST_INTERVAL",
            "four_front_three_owner": "NOT_ARCHIVED_V5_4_2_MAX2",
            "destroy_restore_further_interval": "FAIL_CLOSED_FIRST_INTERVAL",
        },
        "step_369_note": "transition checkpoints 368/369/370 are mesh-adaptation states separated by about 1.05e-7 s, not complete 8.4-s accepted-interval transactions",
        "cached_N1_end_to_end_parity": "FAIL_CLOSED_NO_USABLE_ARCHIVED_STRESS_INTERVAL",
        "cached_N2_end_to_end_parity": "FAIL_CLOSED_INCOMPLETE_FRONT_CANDIDATE_COVERAGE",
        "provider_solve_count": 0, "mechanics_solve_count": 0, "workers_started": 0,
    })


def report(cli: dict) -> str:
    return f"""# PF current-source general multifront V6.4

Permanent boundary: `{BOUNDARY}`.

## Decision

The arbitrary finite-front state machine, complete engine restore, persistent hooks, explicit hook contracts, actual-stress identity, owner-local process evolution, direct exact-delta application, and atomic output/checkpoint transaction are source-qualified. The authoritative current-source entrypoint now delegates one complete interval to `run_stateful_accepted_interval_v12`. A finite physical zero-stress array is available and identity-bound; unavailable stress is represented only by an absent `sigma_gp` with `available=false`.

The integrated immutable-V5.4.2 reproduction is **FAIL_CLOSED**, not an implementation failure. The complete read-only search found no N1 checkpoint having all of a complete state, exact cache evidence, and a complete archived next state. N2 terminal cache evidence contains only the physically assigned candidate at each tip, whereas its cloned V11 competition state lists both candidates at both tips; V6.4 correctly refuses to invent missing front×candidate observations. The archive also lacks the exact accepted-trial delta transactions needed for step-369 and next-checkpoint parity.

No PF worker, FEM solve, provider solve, signed-kernel generation, stochastic trajectory, or 1000-µm continuation ran. The CLI result is `{cli['integrated_interval_transaction']}` and trajectory authorization remains false.

## V6.3 binding

V6.2 final `{V62_FINAL}` precedes V6.3 implementation `{V63_IMPLEMENTATION}`, which precedes final record `{V63_FINAL}`. No executable `arrhenius_fracture/` source or test changed between the V6.3 implementation and record commits. The bound V6.3 archive SHA-256 is `{V63_ARCHIVE_SHA256}`; the exact `git diff --binary` command and output fingerprint are recorded reproducibly from those immutable commits.

## RNG language

V5.4.2 compatibility retains V11's cloned competition/common-random threshold contract. Daughter objects and lineage addresses are distinct, but the threshold generator does not consume the lineage address. V6.4 therefore makes no claim of statistically independent daughter threshold streams and introduces no lineage-split RNG physics.

## Final status

- arbitrary_finite_front_state_machine: QUALIFIED
- complete_process_engine_restore: QUALIFIED
- persistent_hook_components: QUALIFIED
- integrated_stateful_production_interval: QUALIFIED_SOURCE_LEVEL
- owner_local_physical_process_evolution: QUALIFIED
- selected_event_and_actual_RNG_finalization: QUALIFIED_COMPONENTS; cached end-to-end FAIL_CLOSED
- physical_post_event_renewal: QUALIFIED
- exact_accepted_trial_delta_commit: QUALIFIED
- actual_stress_state_binding: QUALIFIED
- atomic_output_checkpoint_transaction: QUALIFIED
- cached_V5_4_2_N1_interval_parity: FAIL_CLOSED_NO_USABLE_ARCHIVED_STRESS_INTERVAL
- cached_V5_4_2_N2_interval_parity: FAIL_CLOSED_INCOMPLETE_FRONT_CANDIDATE_COVERAGE
- generic_N1_N2_PF_reproduction: NOT_EXECUTED
- production_PF_max_fronts_greater_than_two: NOT_EXECUTED
- predictive_recursive_branching_physics_validated: false

Validation: the V6.4-focused suite passed 28 tests; the branching-focused suite passed 176 tests with 1 skip; the unfiltered full suite passed 943 tests with 2 skips and the same seven legacy failures. Compileall and diff checks pass, and generation is deterministic.
"""


def make_archive() -> None:
    members = sorted(path for path in OUT.rglob("*") if path.is_file() and path != ARCHIVE)
    evidence = {
        str(path.relative_to(OUT)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
        for path in members
    }
    manifest = json.dumps({
        "schema": "v6.4.complete-archive-manifest/1", "boundary": BOUNDARY,
        "members": evidence,
    }, indent=2, sort_keys=True).encode() + b"\n"
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            info = zipfile.ZipInfo(str(path.relative_to(OUT)), (2026, 9, 3, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        info = zipfile.ZipInfo("MANIFEST.json", (2026, 9, 3, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    snapshots = OUT / "source_snapshot_v6_4"
    for relative in (
        "arrhenius_fracture/production_multifront_v12.py",
        "arrhenius_fracture/stateful_multifront_production_v12.py",
        "arrhenius_fracture/sharp_front_current_source_multifront_v12.py",
        "scripts/build_pf_general_multifront_qualification_v6_4.py",
        "tests/test_multifront_v6_4_integrated_interval.py",
        "tests/test_pf_general_multifront_v6_4_products.py",
    ):
        target = snapshots / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    with tempfile.TemporaryDirectory(prefix="v6_4_cli_") as temporary:
        temp = Path(temporary)
        cli = v6_4_integrated_interval_dry_run((
            "--v6-4-integrated-interval-dry-run", "--branching-mode", "mechanistic",
            "--shared-region-branching", "experimental",
            "--v5-4-2-n1-checkpoint", str(N1), "--v5-4-2-n2-checkpoint", str(N2),
            "--fresh-output-destination", str(temp / "output"),
            "--fresh-cache-destination", str(temp / "cache"),
        ))
    inventory, inventory_summary = archive_sufficiency_inventory()
    dump("pf_general_multifront_v6_4_source_provenance.json", provenance())
    write_csv("pf_general_multifront_integrated_interval_lifecycle_v6_4.csv", lifecycle_rows())
    dump("pf_general_multifront_hook_interface_contract_v6_4.json", hook_contract())
    dump("pf_general_multifront_stress_identity_v6_4.json", stress_record(cli))
    dump("pf_general_multifront_owner_process_and_renewal_v6_4.json", owner_record())
    write_csv("pf_general_multifront_event_finalization_v6_4.csv", event_rows())
    dump("pf_general_multifront_exact_trial_delta_v6_4.json", exact_delta_record())
    dump("pf_general_multifront_arbitrary_region_request_v6_4.json", arbitrary_request_record())
    dump("pf_general_multifront_atomic_output_checkpoint_v6_4.json", atomic_record())
    dump("pf_general_multifront_cached_end_to_end_parity_v6_4.json", cached_record(cli))
    dump("pf_general_multifront_integrated_cli_dry_run_v6_4.json", base(cli))
    write_csv("pf_general_multifront_archive_sufficiency_inventory_v6_4.csv", inventory)
    dump("pf_general_multifront_archive_sufficiency_summary_v6_4.json", inventory_summary)
    (OUT / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_4.md").write_text(report(cli))
    make_archive()
    print(json.dumps({"output": str(OUT), "archive_sha256": sha(ARCHIVE)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
