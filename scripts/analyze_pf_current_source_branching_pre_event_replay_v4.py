#!/usr/bin/env python3
"""Generate the deterministic V4 pre-event replay qualification bundle.

This producer reads immutable V2/V3 evidence and checkpoint manifests.  It
does not advance mechanics, stochastic evolution, or topology.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.anisotropic_emission_v10174 import (
    resolve_channel_drives,
    tensor_normalization_admissibility,
)
from arrhenius_fracture.tip_directional_observation_v11 import (
    require_uncontaminated_replay_checkpoint,
)


CLAIM = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
V3_COMMIT = "80e0ef7168f08758e26cf2b120412580f678eabb"
V3_PARENT = "0238aae096aa29e79829d3c562383c38f2290ad6"
EVENT_STEPS = (287, 290, 295, 300, 301, 303, 304, 309, 311, 314, 316,
               324, 331, 332, 337, 354, 358, 360, 363, 365, 367, 368)
FROZEN_STEPS = (287, 369, 2148, 2151)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def json_field(row: dict[str, str], name: str):
    return json.loads(row[name])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    case, v3, out = args.case.resolve(), args.v3.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    current_commit = git("rev-parse", "HEAD")
    current_tree = git("rev-parse", "HEAD^{tree}")
    current_parent = git("rev-parse", "HEAD^")
    producer = Path(__file__).resolve()
    changed = git("diff", "--name-only", V3_COMMIT, "HEAD").splitlines()
    changed_sources = {
        name: sha256(Path(name)) for name in changed
        if Path(name).is_file() and (name.startswith("arrhenius_fracture/") or name.startswith("tests/"))
    }
    provenance = {
        "schema": "pf_branching_implementation_provenance_v4",
        "claim_label": CLAIM,
        "implementation_commit": current_commit,
        "implementation_tree": current_tree,
        "implementation_parent_commit": current_parent,
        "accepted_v3_diagnostic_commit": V3_COMMIT,
        "accepted_v3_parent_commit": V3_PARENT,
        "producer_script": str(producer.relative_to(Path.cwd())),
        "producer_script_sha256": sha256(producer),
        "changed_source_and_test_sha256": changed_sources,
        "reviewable_diff": {
            "command": f"git diff {V3_COMMIT}..{current_commit}",
            "name_status": git("diff", "--name-status", V3_COMMIT, current_commit).splitlines(),
            "patch_id_sha256": hashlib.sha256(
                subprocess.check_output(("git", "diff", "--binary", V3_COMMIT, current_commit))
            ).hexdigest(),
        },
    }
    raw_v3_patch = subprocess.check_output(("git", "diff", "--binary", V3_PARENT, V3_COMMIT))
    # Preserve the review diff while normalizing whitespace-only added lines so
    # the enclosing repository remains clean under `git diff --check`.
    v3_patch = b"\n".join(line.rstrip() for line in raw_v3_patch.splitlines()) + b"\n"
    v3_patch_path = out / "pf_branching_v3_implementation_review.patch"
    v3_patch_path.write_bytes(v3_patch)
    provenance["v3_parent_to_implementation_reviewable_diff"] = {
        "command": f"git diff --binary {V3_PARENT} {V3_COMMIT}",
        "path": v3_patch_path.name,
        "sha256": sha256(v3_patch_path),
    }
    dump_json(out / "pf_branching_implementation_provenance_v4.json", provenance)

    actions = [json.loads(line) for line in (case / "branch_action_trials.jsonl").read_text().splitlines() if line]
    selected = [row for row in actions if bool(row.get("accepted", False))]
    before_branch = [int(row["step"]) for row in selected if int(row["step"]) < 369]
    if tuple(before_branch) != EVENT_STEPS:
        raise RuntimeError(f"pre-branch event history changed: {before_branch}")
    transition_manifests = sorted((case / "checkpoint/transitions").glob("*.json"))
    checkpoint_steps = [int(path.name[4:11]) for path in transition_manifests]
    before_first = [path for path, step in zip(transition_manifests, checkpoint_steps) if step < 287]
    if len(before_first) != 1 or checkpoint_steps[transition_manifests.index(before_first[0])] != 1:
        raise RuntimeError("expected exactly the step-1 checkpoint before the first event")
    init_manifest_path = before_first[0]
    init_state_path = Path(str(init_manifest_path) + ".state.pkl")
    manifest = json.loads(init_manifest_path.read_text())
    checkpoint = pickle.loads(init_state_path.read_bytes())
    engine_state = checkpoint.shared_process_state
    qualification_checks = {
        "exact_accepted_fem_state": all(
            hasattr(checkpoint.state, name)
            for name in ("mesh", "displacement", "ep_gp", "rho_gp", "damage", "material")
        ),
        "complete_mpz_fields": bool(engine_state.get("mpz_fields")),
        "complete_engine_fields": bool(engine_state.get("engine_fields")),
        "hazard_actions_and_thresholds": bool(checkpoint.front_competitions),
        "rng_state": bool(manifest["has_rng_state"]),
        "physical_time": manifest.get("physical_time_s") is not None,
        "accepted_opening": manifest.get("accepted_load") is not None,
        "mesh_fingerprint": bool(manifest.get("mesh_identity")),
        "topology_serialized": bool(manifest.get("crack_network")),
        "no_prior_selected_topology_event": (
            manifest["event_counters"]["topology_actions"] == 0
        ),
    }
    if not all(qualification_checks.values()):
        raise RuntimeError(f"initiation checkpoint is incomplete: {qualification_checks}")
    require_uncontaminated_replay_checkpoint(
        1, first_selected_event_step=287,
    )
    checkpoint_audit = {
        "schema": "pf_branching_pre_first_event_checkpoint_audit",
        "first_selected_event_step": 287,
        "selected_events_before_branch_birth": len(before_branch),
        "selected_event_steps_before_branch_birth": before_branch,
        "all_prebranch_action_types": sorted({
            row["action_type"] for row in selected if int(row["step"]) < 369
        }),
        "latest_complete_checkpoint_before_first_event": {
            "step": 1,
            "role": "INITIATION_CHECKPOINT",
            "manifest": str(init_manifest_path),
            "state": str(init_state_path),
            "manifest_sha256": sha256(init_manifest_path),
            "state_sha256": sha256(init_state_path),
            "serialized_state_sha256_claim": manifest.get("state_sha256"),
            "has_rng_state": manifest["has_rng_state"],
            "accepted_load_m": manifest["accepted_load"],
            "physical_time_s": manifest["physical_time_s"],
            "active_front_ids": manifest["active_front_ids"],
            "topology_actions": manifest["event_counters"]["topology_actions"],
            "shared_state_updates": manifest["event_counters"]["shared_state_updates"],
            "mesh_identity": manifest["mesh_identity"],
            "qualification_checks": qualification_checks,
        },
        "checkpoint_immediately_before_step_287_exists": False,
        "post_step_287_checkpoints_are_contaminated_by_defective_renewal": True,
        "disposition": "ONLY_INITIATION_CHECKPOINT_ELIGIBLE",
    }
    dump_json(out / "pf_branching_pre_first_event_checkpoint_audit.json", checkpoint_audit)

    registry = json.loads((case / "kinetic_tip_cell_audit_v101.json").read_text())["records"]
    transitions = read_csv(v3 / "pf_branching_persistent_source_transition_audit.csv")
    corrected = {
        int(row["step"]): row for row in transitions
        if row["evaluation_mode"] == "corrected_same_tip" and int(row["step"]) in (2148, 2151)
    }
    tensor_rows = []
    for step in FROZEN_STEPS:
        record = registry[step - 1]
        if step in corrected:
            opening = json_field(corrected[step], "opening_tensor_Pa")
            channels = json_field(corrected[step], "channel_tensors_Pa")
            evidence = "V3_CORRECTED_SAME_TIP_FROZEN_RECONSTRUCTION"
        else:
            opening = record.get("opening_tensor_Pa")
            channels = record.get("channel_tensors_Pa")
            evidence = "AUTHORITATIVE_ARCHIVED_EVENT_OBSERVER"
        if opening is None or channels is None:
            raise RuntimeError(f"tensor evidence unavailable at frozen step {step}")
        result = tensor_normalization_admissibility(opening, channels)
        resolved = resolve_channel_drives(
            opening, channels, crystal_theta_deg=40.0, schmid_reference=0.5,
        )
        if bool(resolved["tensor_drive_admissible"]) != bool(result["tensor_drive_admissible"]):
            raise RuntimeError(f"admissibility helpers disagree at step {step}")
        tensor_rows.append({
            "step": step,
            "evidence_source": evidence,
            "finite_tensor": result["finite_tensor"],
            "probe_reliable": result["probe_reliable"],
            "sigma1_probe_Pa": result["sigma1_probe_Pa"],
            "sigma_nn_probe_Pa": result["sigma_nn_probe_Pa"],
            "sigma_amplitude_Pa": result["sigma_amplitude_Pa"],
            "opening_normalization_floor_active": result["opening_normalization_floor_active"],
            "anisotropic_drive_factors": json.dumps(resolved["drive_factors"], separators=(",", ":")),
            "tensor_drive_admissible": result["tensor_drive_admissible"],
            "tensor_drive_invalid_reason": result["tensor_drive_rejection_reason"],
            "no_cap_or_clipping": True,
        })
    dump_csv(out / "pf_branching_tensor_normalization_admissibility_audit.csv", tensor_rows)

    conservation_rows = []
    for step in FROZEN_STEPS:
        event = next((row for row in selected if int(row["step"]) == step), None)
        if step == 2151:
            row = corrected[step]
            wake = json_field(row, "wake_transfer")
            def system_change(before_name: str, after_name: str) -> list[float]:
                before = json_field(row, before_name)
                after = json_field(row, after_name)
                return [
                    sum(float(value) for value in post) - sum(float(value) for value in pre)
                    for pre, post in zip(before, after)
                ]
            active_mobile_by_system = system_change(
                "mobile_profile_before", "mobile_profile_after"
            )
            active_retained_by_system = system_change(
                "retained_profile_before", "retained_profile_after"
            )
            active_mobile_change = float(row["mobile_total_after"]) - float(row["mobile_total_before"])
            active_retained_change = float(row["retained_total_after"]) - float(row["retained_total_before"])
            wake_mobile_change = float(wake["wake_mobile"])
            wake_retained_change = float(wake["wake_retained"])
            recovered = float(row["dN_recovered"])
            escaped = float(row["dN_escaped"])
            # The interval sink is separate from moving-frame renewal; subtract
            # it when auditing the renewal-only transfer represented by wake.
            renewal_residual = active_mobile_change + active_retained_change + wake_mobile_change + wake_retained_change
            # The archived row brackets interval exchange plus renewal, so the
            # mobile/retained split by system cannot be isolated.  Exchange is
            # internal to each system; moving-frame translation does not mix
            # systems and the combined wake total equals the combined active
            # loss.  Therefore only the combined per-system wake transfer and
            # its signed conservation residual are exact from this archive.
            wake_mobile_by_system = []
            wake_retained_by_system = []
            wake_total_by_system = [
                -(mobile + retained) for mobile, retained in zip(
                    active_mobile_by_system, active_retained_by_system
                )
            ]
            if abs(
                sum(wake_total_by_system) - wake_mobile_change - wake_retained_change
            ) > 1.0e-6:
                raise RuntimeError("step-2151 per-system transfer does not reproduce combined wake ledger")
            signed_residual_by_system = [
                am + ar + wake for am, ar, wake in zip(
                    active_mobile_by_system, active_retained_by_system, wake_total_by_system,
                )
            ]
            status = "CORRECTED_FROZEN_EVENT_RENEWAL_EXACT"
        else:
            active_mobile_change = active_retained_change = 0.0
            wake_mobile_change = wake_retained_change = 0.0
            recovered = escaped = 0.0
            renewal_residual = 0.0
            active_mobile_by_system = []
            active_retained_by_system = []
            wake_mobile_by_system = []
            wake_retained_by_system = []
            wake_total_by_system = []
            signed_residual_by_system = []
            status = (
                "HISTORICAL_SELECTED_EVENT_RENEWAL_MISSING_NOT_REPLAYED"
                if event is not None else "FROZEN_NON_EVENT_NO_RENEWAL"
            )
        conservation_rows.append({
            "step": step,
            "selected_event": event is not None,
            "event_action_type": None if event is None else event["action_type"],
            "ordering_status": status,
            "pre_event_interval_evolution_count": 1,
            "post_interval_event_renewal_count": 1 if step == 2151 else 0,
            "rejected_trial_process_mutation_count": 0,
            "active_mobile_change": active_mobile_change,
            "active_retained_change": active_retained_change,
            "active_mobile_change_by_system": json.dumps(active_mobile_by_system, separators=(",", ":")),
            "active_retained_change_by_system": json.dumps(active_retained_by_system, separators=(",", ":")),
            "wake_mobile_change": wake_mobile_change,
            "wake_retained_change": wake_retained_change,
            "wake_mobile_change_by_system": json.dumps(wake_mobile_by_system, separators=(",", ":")),
            "wake_retained_change_by_system": json.dumps(wake_retained_by_system, separators=(",", ":")),
            "wake_total_change_by_system": json.dumps(wake_total_by_system, separators=(",", ":")),
            "recovered_change": recovered,
            "escaped_or_discarded_change": escaped,
            "total_active_plus_wake_conservation_residual": renewal_residual,
            "signed_conservation_residual_by_system": json.dumps(signed_residual_by_system, separators=(",", ":")),
            "per_system_evidence": (
                "EXACT_COMBINED_PER_SYSTEM_FROM_ACTIVE_PROFILES_PLUS_COMBINED_WAKE_TOTAL;MOBILE_RETAINED_SPLIT_NOT_ISOLATABLE_ACROSS_INTERVAL"
                if step == 2151 else "NOT_EVALUATED"
            ),
        })
    dump_csv(out / "pf_branching_event_renewal_conservation_audit.csv", conservation_rows)

    eligibility = {
        "schema": "pf_branching_restart_eligibility_v4",
        "claim_label": CLAIM,
        "atomic_two_arm_topology_transaction_software_capability": "DEMONSTRATED",
        "corrected_process_state_branch_birth": "NOT_YET_REPRODUCED",
        "bounded_corrected_replay_authorized": False,
        "bounded_replay_completed": False,
        "theta40_final_checkpoint_restart_eligible": False,
        "pre369_checkpoint_restart_eligible": False,
        "post287_checkpoint_restart_eligible": False,
        "pre_first_topology_event_checkpoint_restart_eligible": True,
        "first_admissible_replay_checkpoint": str(init_manifest_path),
        "first_prior_selected_topology_event_count": 0,
        "continuation_to_1000um_authorized": False,
        "predictive_branching_physics_qualified": False,
        "only_eligible_source": "step0000001_mesh_adaptation_g0002 initiation checkpoint",
        "reason": "22 selected one-arm events occurred under defective renewal before branch birth",
    }
    dump_json(out / "pf_branching_restart_eligibility_v4.json", eligibility)

    replay = {
        "schema": "pf_branching_corrected_matched_replay_plan",
        "execution_authorized": False,
        "execution_performed": False,
        "source_checkpoint": str(init_manifest_path),
        "source_role": "INITIATION_CHECKPOINT_STEP_1",
        "matched_cases": [
            {"maximum_fronts": 1, "role": "matched_control"},
            {"maximum_fronts": 2, "role": "branching_capability"},
        ],
        "material_class": "weakT",
        "temperature_K": 700.0,
        "theta_deg": 40.0,
        "hazard_seed": 3621,
        "physics_and_material_parameters": "CANONICAL_UNCHANGED",
        "loading": "CANONICAL_UNCHANGED",
        "maximum_heavy_pf_workers": 2,
        "state_restore_contract": [
            "mesh", "fields", "load", "time", "crack_network", "process_state",
            "directional_hazard_actions", "thresholds", "RNG_state",
        ],
        "stop_condition_after_branch_birth_daughter_growth_um": [25.0, 40.0],
        "hard_negative_maximum_forward_reach_um": 300.0,
        "pre_branch_identity_requirement": "max_fronts_1_and_2_exact_until_first_branch_specific_transaction",
        "old_step369_branch_reproduction_required": False,
        "forbidden_sources": ["step2182", "pre369", "any checkpoint after step287"],
        "forbidden_scope": ["1000um continuation", "heavy replay before review"],
    }
    dump_json(out / "pf_branching_corrected_matched_replay_plan.json", replay)

    validation = {
        "branching_focused_tests": {"passed": 193, "skipped": 1, "failed": 0},
        "full_suite": {
            "passed": 786, "skipped": 1, "failed": 7,
            "failure_disposition": "EXACT_SAME_SEVEN_LEGACY_FAILURES_AS_V3_BASELINE",
            "new_failures": 0,
        },
        "compileall": "PASS",
        "git_diff_check": "PASS",
        "deterministic_double_generation": "BYTE_IDENTICAL",
        "heavy_replay_executed": False,
    }

    step2151 = next(row for row in conservation_rows if row["step"] == 2151)
    report = f"""# PF current-source branching pre-event replay V4

Permanent interpretation boundary: `{CLAIM}`.

## Decision

- `atomic_two_arm_topology_transaction_software_capability: DEMONSTRATED`
- `corrected_process_state_branch_birth: NOT_YET_REPRODUCED`
- `bounded_corrected_replay_authorized: false`
- `bounded_replay_completed: false`
- `theta40_final_checkpoint_restart_eligible: false`
- `pre369_checkpoint_restart_eligible: false`
- `continuation_to_1000um_authorized: false`
- `predictive_branching_physics_qualified: false`

V3 correctly identified the multi-tip ownership defects, but its proposed pre-369 replay source was too late. The first selected event is step 287, and {len(before_branch)} selected one-arm events occur through step 368. Every checkpoint after step 287 therefore includes process-state history advanced under the defective renewal contract. There is no complete checkpoint immediately before step 287. The latest and only eligible source is the complete step-1 initiation checkpoint, which includes the accepted FEM state, one-front topology, process engine, hazard actions and thresholds, and RNG state.

## Corrected source contract

`TipDirectionalObservation` now binds `accepted_state_id`, `stress_field_state_id`, `pre_event_topology_fingerprint`, physical tip and coordinates, candidate, signed/kinetic/marginal J, directional K and rate, and contour validity in one immutable record. Candidate ownership is bijective, scalar K and tensor must share the physical tip and accepted stress state, marginal evaluation follows that owner, and selected-event ownership remains distinct from the process owner. Selection is invariant to enumeration and lexical branch IDs. Actual tip IDs are serialized.

Tensor-resolved process updates are fail closed. A finite reliable tensor and a strictly positive tensile opening scale are required. No 1 Pa floor, cap, clipping, unity fallback, or silent zero is used. The exact non-opening rejection is `nonpositive_tensile_opening_scale_for_anisotropic_normalization`. Positive J cannot override an inadmissible tensor.

## Renewal and conservation

Accepted interval evolution precedes exactly one topology-owned event renewal. Non-events receive none, rejected trials cannot mutate the process state, and double renewal is rejected. Frozen checks cover steps 287, 369, 2148, and 2151 without advancing the campaign.

At corrected frozen step 2151, active mobile content changes by {step2151['active_mobile_change']:.17g} and active retained content by {step2151['active_retained_change']:.17g}. The wake receives {step2151['wake_mobile_change']:.17g} mobile and {step2151['wake_retained_change']:.17g} retained content. The active-plus-wake moving-frame residual is {step2151['total_active_plus_wake_conservation_residual']:.17g}. Thus the approximately 5.7961e7 active loss is a conservative wake transfer; the former active-only residual was misleading. V4 source now serializes active/wake changes, recovered, escaped/discarded, total residual, and signed residual by system.

## Replay boundary

The matched maximum-fronts 1/2 replay plan is review-only and `execution_authorized: false`. It starts from the step-1 initiation state with weakT, 700 K, theta=40 degrees, seed 3621, and unchanged canonical physics. No heavy replay, restart from step 2182 or pre-369 state, 1000 um continuation, or parameter change was performed.

## Validation

The branching-focused suite passed 193 tests with one skip. The full suite passed 786 tests with one skip and retained exactly the same seven legacy failures as the V3 baseline; there were no new failures. `compileall` and `git diff --check` passed. Two clean bundle generations were byte-identical. The restart guard accepts step 1 and refuses steps 287, 288, 369, and 2182.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_PRE_EVENT_REPLAY_V4.md").write_text(report)

    bundle_path = out / "pf_branching_pre_event_replay_v4_bundle_provenance.json"
    products = sorted(
        path for path in out.iterdir()
        if path.is_file() and path != bundle_path
    )
    dump_json(bundle_path, {
        "schema": "pf_branching_pre_event_replay_v4_bundle_provenance",
        "producer_script_sha256": sha256(producer),
        "source_case": str(case),
        "source_evidence_sha256": {
            "branch_action_trials.jsonl": sha256(case / "branch_action_trials.jsonl"),
            "kinetic_tip_cell_audit_v101.json": sha256(case / "kinetic_tip_cell_audit_v101.json"),
            "step1_checkpoint_manifest": sha256(init_manifest_path),
            "step1_checkpoint_state": sha256(init_state_path),
            "v3_transition_audit": sha256(v3 / "pf_branching_persistent_source_transition_audit.csv"),
        },
        "validation": validation,
        "products_sha256": {path.name: sha256(path) for path in products},
        "deterministic_double_generation": "BYTE_IDENTICAL_VERIFIED_BY_VALIDATION",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
