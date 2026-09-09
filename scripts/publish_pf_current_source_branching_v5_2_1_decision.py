#!/usr/bin/env python3
"""Publish the analysis-only V5.2.1 morphology and coverage correction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
PREDECESSOR_COMMIT = "ff6bc653298b13f6da12c6b5722684b9f115afde"
EXECUTION_COMMIT = "e2aff736afe0e1d2d1b600c25743de317a71c7ba"
EXECUTION_TREE = "14a953037a44a25288c41d317a0ae0c7f36d22bc"
RAW_TREE_SHA256 = "cec0e2523bd16ce18b541c1eb7cdf65ee26ba553b5ecfb657da617ed2321565e"
INITIAL_TIP_X_UM = 500.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def strict_below_target_events(gap_um: float, projection_um: float) -> int:
    """Number of fixed events that can occur while the tip remains below target."""
    if gap_um <= 0.0 or projection_um <= 0.0:
        return 0
    return max(0, math.ceil(gap_um / projection_um - 1.0e-13) - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-result", type=Path, required=True)
    parser.add_argument("--old-coverage-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.v5_result.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    old_decision_path = source / "final_two_axis_decision.json"
    old_decision = read_json(old_decision_path)
    topology = read_json(source / "topology_wake_rng_geometry_audit.json")
    prebranch = read_json(source / "prebranch_identity_audit.json")
    ownership = read_json(source / "scalar_tensor_ownership_summary.json")
    renewal = read_csv(source / "process_state_and_renewal_audit.csv")
    terminal_state = {row["case"]: row for row in read_csv(source / "terminal_process_state.csv")}
    failures = read_json(source / "terminal_failure_summary.json")
    enabled_actions = read_jsonl(source / "enabled_max2_branch_action_trials.jsonl")
    control_actions = read_jsonl(source / "control_max1_branch_action_trials.jsonl")
    branch_events = read_csv(source / "enabled_max2_branch_events.csv")
    cluster_rows = read_csv(source / "enabled_max2_branch_clusters.csv")
    enabled_fronts = read_csv(source / "enabled_max2_fronts.csv")
    enabled_status = read_json(source / "enabled_max2_worker_status.json")
    raw_freeze = read_json(source / "raw_tree_freeze_manifest.json")
    old_coverage = read_json(args.old_coverage_audit.resolve())

    if old_decision["pair_terminal_result"] != "CORRECTED_THETA40_REPLAY_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE":
        raise RuntimeError("unexpected predecessor terminal decision")
    if raw_freeze["tree_sha256"] != RAW_TREE_SHA256:
        raise RuntimeError("unexpected V5.2 raw-tree fingerprint")
    if enabled_status["execution_commit"] != EXECUTION_COMMIT or enabled_status["execution_tree"] != EXECUTION_TREE:
        raise RuntimeError("unexpected promoted execution source")

    births = [row for row in enabled_actions if row["accepted"] and row["action_type"] == "two_arm"]
    if len(births) != 1:
        raise RuntimeError("expected exactly one accepted two-arm birth")
    birth = births[0]
    final_cluster = cluster_rows[-1]
    renewal_by_case = {row["case"]: row for row in renewal}
    ledger_pass = bool(topology["renewal_and_geometry_closure_pass"] and topology["population_and_signed_ledgers_pass"])

    criteria = [
        {
            "criterion": 1,
            "name": "directional_first_passage",
            "pass": len(birth["completion_times_s"]) == 2 and all(math.isfinite(float(value)) for value in birth["completion_times_s"]),
            "evidence": {"step": birth["step"], "completion_times_s": birth["completion_times_s"], "correlation_time_difference_s": birth["correlation_time_difference_s"], "pending_event_ids": birth["pending_event_ids"]},
        },
        {
            "criterion": 2,
            "name": "positive_signed_directional_J",
            "pass": all(float(value) > 0.0 for value in birth["signed_directional_J_J_per_m2"]),
            "evidence": {"signed_directional_J_J_per_m2": birth["signed_directional_J_J_per_m2"], "local_J_valid": birth["local_J_valid"]},
        },
        {
            "criterion": 3,
            "name": "energy_acceptance",
            "pass": bool(birth["accepted"] and float(birth["net_energy_margin_J_per_m"]) > 0.0),
            "evidence": {"released_energy_J_per_m": birth["released_energy_J_per_m"], "total_cost_J_per_m": birth["total_dissipative_cost_J_per_m"], "net_energy_margin_J_per_m": birth["net_energy_margin_J_per_m"]},
        },
        {
            "criterion": 4,
            "name": "valid_daughter_topology",
            "pass": topology["accepted_two_arm_birth_count"] == 1 and len(topology["daughter_growth_um"]) == 2,
            "evidence": {"accepted_two_arm_birth_count": topology["accepted_two_arm_birth_count"], "daughter_ids": sorted(topology["daughter_growth_um"])},
        },
        {
            "criterion": 5,
            "name": "nonstub_propagation",
            "pass": bool(topology["committed_nonstub_daughter_observed"]),
            "evidence": {"daughter_growth_um": topology["daughter_growth_um"], "daughter_event_counts": topology["daughter_event_counts"]},
        },
        {
            "criterion": 6,
            "name": "cluster_bookkeeping",
            "pass": bool(topology["cluster_bookkeeping_valid"] and topology["final_cluster_unresolved"]),
            "evidence": {"cluster_bookkeeping_valid": topology["cluster_bookkeeping_valid"], "final_cluster_unresolved": topology["final_cluster_unresolved"]},
        },
        {
            "criterion": 7,
            "name": "conditional_handoff",
            "pass": topology["final_cluster_handoff_required"] is False,
            "evidence": {"handoff_required_at_stop": topology["final_cluster_handoff_required"], "independent_local_J_flags": topology["final_independent_local_J_flags"], "short_arm_growth_um": min(topology["daughter_growth_um"].values()), "handoff_length_um": float(final_cluster["branch_handoff_length_m"]) * 1e6, "short_arm_below_handoff_length": min(topology["daughter_growth_um"].values()) < float(final_cluster["branch_handoff_length_m"]) * 1e6},
        },
        {
            "criterion": 8,
            "name": "no_front_cap_binding",
            "pass": bool(topology["no_front_cap_veto_recorded"]),
            "evidence": {"no_front_cap_veto_recorded": topology["no_front_cap_veto_recorded"], "maximum_fronts": enabled_status["maximum_fronts"]},
        },
        {
            "criterion": 9,
            "name": "no_immediate_retirement_or_reversal",
            "pass": bool(topology["no_immediate_daughter_retirement"] and topology["all_committed_segments_forward_in_x"] and topology["no_bridge_or_reconnection_veto_recorded"]),
            "evidence": {"daughter_statuses_at_stop": topology["daughter_statuses_at_stop"], "all_committed_segments_forward_in_x": topology["all_committed_segments_forward_in_x"], "no_bridge_or_reconnection_veto_recorded": topology["no_bridge_or_reconnection_veto_recorded"]},
        },
        {
            "criterion": 10,
            "name": "state_hazard_rng_wake_geometry_ledger_closure",
            "pass": bool(ledger_pass and topology["accepted_hazard_event_ids_unique"] and topology["hazard_rng_state_serialized"] and all(all(values.values()) for values in topology["hazard_event_ordinal_closure"].values())),
            "evidence": {"renewal_and_geometry_closure_pass": topology["renewal_and_geometry_closure_pass"], "population_and_signed_ledgers_pass": topology["population_and_signed_ledgers_pass"], "hazard_event_ordinal_closure": topology["hazard_event_ordinal_closure"], "hazard_rng_state_serialized": topology["hazard_rng_state_serialized"], "one_process_update_per_interval": {case: row["one_process_update_per_interval"] == "True" for case, row in renewal_by_case.items()}},
        },
        {
            "criterion": 11,
            "name": "prebranch_matched_pair_identity",
            "pass": bool(prebranch["pass"]),
            "evidence": {"identical_through_step": prebranch["branch_specific_transaction_step"] - 1, "directional_rows_each": prebranch["directional_rows_each"], "action_rows_each": prebranch["action_rows_each"]},
        },
    ]
    all_criteria_pass = all(item["pass"] for item in criteria)
    if not all_criteria_pass or not ownership["postbranch_contract_pass"]:
        raise RuntimeError("morphology success criteria do not close")

    decision = {
        "schema": "pf_branching_corrected_v5_2_1_superseding_decision_v1",
        "permanent_interpretation_boundary": LABEL,
        "supersedes_only": portable_path(old_decision_path),
        "superseded_decision_sha256": sha256(old_decision_path),
        "immutable_predecessor_result_commit": PREDECESSOR_COMMIT,
        "promoted_execution_source": {"commit": EXECUTION_COMMIT, "tree": EXECUTION_TREE},
        "raw_tree_sha256": RAW_TREE_SHA256,
        "pair_terminal_result": "CORRECTED_THETA40_REPLAY_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE",
        "atomic_topology_transaction_capability": "DEMONSTRATED",
        "corrected_process_state_branch_birth": "DEMONSTRATED",
        "corrected_morphology_capability": "CORRECTED_CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED_BEFORE_ENVELOPE_STOP",
        "bounded_300um_pair_completion": "NOT_COMPLETED_SIGNED_KERNEL_ENVELOPE",
        "final_independent_tip_mechanics": "UNQUALIFIED_ONE_OF_TWO_DAUGHTER_LOCAL_J_CONTOURS_VALID_AT_STOP",
        "cluster_handoff_status": "UNRESOLVED_HANDOFF_NOT_REQUIRED_AT_STOP",
        "predictive_branching_physics_validated": False,
        "morphology_success_criteria": criteria,
        "all_eleven_morphology_success_criteria_pass": all_criteria_pass,
        "additional_process_ownership_gate": {"postbirth_steps": ownership["enabled_steps_at_or_after_branch_birth"], "same_tip_contract_pass": ownership["postbranch_contract_pass"]},
        "endpoint_semantics": "300 micrometres is the stopping target and hard-negative ceiling, not an additional morphology-success gate.",
        "record_scope": "ANALYSIS_ONLY_NO_PF_NO_FEM_NO_PARAMETER_CHANGE",
    }
    dump(out / "pf_branching_corrected_v5_2_1_decision.json", decision)

    target_um = float(old_coverage["target_projected_crack_extension_um"])
    theta_deg = float(old_coverage["fixed_crack_tilt_deg"])
    old_min_path_um = target_um / math.cos(math.radians(theta_deg))
    old_margin_um = float(old_coverage["maximum_event_advance_um"])
    old_required_um = old_min_path_um + old_margin_um
    if not math.isclose(old_required_um, float(old_coverage["required_atlas_max_crack_path_extension_um"]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("old coverage formula does not reproduce")

    event = branch_events[0]
    arm_ids = json.loads(event["arm_front_ids"])
    arm_directions = json.loads(event["arm_directions"])
    candidate_ids = json.loads(event["plane_identities"])
    candidate_inventory = {
        candidate: {"angle_rad": float(angle), "angle_deg": math.degrees(float(angle))}
        for candidate, angle in zip(candidate_ids, arm_directions)
    }
    da_um = float(old_coverage["base_checkpoint_um"])
    for item in candidate_inventory.values():
        item["fixed_event_length_um"] = da_um
        item["positive_x_projection_um"] = da_um * math.cos(item["angle_rad"])
    minimum_projection_um = min(item["positive_x_projection_um"] for item in candidate_inventory.values())
    single_front_below_events = strict_below_target_events(target_um, minimum_projection_um)
    single_front_pre_event_query_um = single_front_below_events * da_um

    accepted_control = [row for row in control_actions if row["accepted"]]
    control_counts = Counter(row["candidate_ids"][0] for row in accepted_control)
    control_projected_um = float(terminal_state["control_max1"]["projected_extension_um"])
    next_reaches = {
        candidate: control_projected_um + item["positive_x_projection_um"]
        for candidate, item in candidate_inventory.items()
    }
    coverage_failure = {
        "schema": "pf_branching_signed_kernel_coverage_failure_audit_v5_2_1",
        "old_coverage_audit": portable_path(args.old_coverage_audit),
        "old_coverage_audit_sha256": sha256(args.old_coverage_audit.resolve()),
        "old_planner_reproduction": {
            "target_projected_um": target_um,
            "fixed_path_angle_deg": theta_deg,
            "target_divided_by_cos_theta_um": old_min_path_um,
            "legacy_threshold_scaled_maximum_event_margin_um": old_margin_um,
            "required_um": old_required_um,
            "measured_endpoint_um": float(old_coverage["atlas_max_crack_path_extension_um"]),
            "coverage_margin_um": float(old_coverage["coverage_margin_um"]),
            "formula_reproduced": True,
        },
        "failure_cause": [
            "The old planner assumed one fixed +40 degree path instead of the actual +40/-50 candidate inventory.",
            "The old planner used the inherited threshold-scaled event reward instead of the corrected fixed 5 micrometre topology arm length.",
            "For multiple active fronts, shared process extension is not maximum forward reach and can increase when a nonleading front advances.",
        ],
        "corrected_event_contract": {"topology_owner": "V11 directional topology", "fixed_realized_arm_length_um": da_um, "legacy_threshold_scaled_setting_retained_as_engine_provenance_only": True},
        "candidate_inventory": candidate_inventory,
        "corrected_single_front_semantics": {
            "minimum_positive_candidate_projection_um": minimum_projection_um,
            "maximum_events_while_strictly_below_300um": single_front_below_events,
            "maximum_pre_event_shared_extension_query_um": single_front_pre_event_query_um,
            "one_topology_quantum_guard_endpoint_um": single_front_pre_event_query_um + da_um,
            "scope": "conservative single-front bound for this exact candidate inventory",
        },
        "observed_control": {
            "accepted_event_count": len(accepted_control),
            "accepted_events_by_candidate": dict(control_counts),
            "accepted_events_by_angle_deg": {str(round(candidate_inventory[candidate]["angle_deg"])): count for candidate, count in control_counts.items()},
            "shared_process_extension_um": float(terminal_state["control_max1"]["shared_process_physical_advance_um"]),
            "projected_extension_um": control_projected_um,
            "failed_query_um": float(failures["control_max1"]["attempted_query_m"]) * 1e6,
            "qualified_family_maximum_um": float(failures["control_max1"]["qualified_maximum_m"]) * 1e6,
            "next_projected_reach_um_by_candidate": next_reaches,
            "either_permitted_next_event_crosses_target": all(value >= target_um for value in next_reaches.values()),
        },
        "coverage_disposition": "PLANNING_ERROR_NOT_FRACTURE_OR_BRANCHING_FAILURE",
        "no_new_kernel_calculation_performed": True,
    }
    dump(out / "pf_branching_signed_kernel_coverage_failure_audit.json", coverage_failure)

    final_step = max(int(row["step"]) for row in enabled_fronts)
    final_fronts = [row for row in enabled_fronts if int(row["step"]) == final_step]
    active = {row["front_id"]: row for row in final_fronts if row["status"] == "active"}
    if set(active) != set(arm_ids):
        raise RuntimeError("terminal active fronts do not match branch daughters")
    target_x_um = INITIAL_TIP_X_UM + target_um
    per_front = {}
    for front_id, angle in zip(arm_ids, arm_directions):
        row = active[front_id]
        tip_x_um = float(row["tip_x_m"]) * 1e6
        projection_um = da_um * math.cos(float(angle))
        gap_um = target_x_um - tip_x_um
        per_front[front_id] = {
            "angle_deg": math.degrees(float(angle)),
            "tip_x_um": tip_x_um,
            "projected_extension_um": tip_x_um - INITIAL_TIP_X_UM,
            "projected_gap_to_target_um": gap_um,
            "one_event_positive_x_projection_um": projection_um,
            "additional_events_possible_while_remaining_strictly_below_target": strict_below_target_events(gap_um, projection_um),
        }
    sorted_fronts = sorted(per_front.items(), key=lambda pair: pair[1]["projected_extension_um"])
    short_id, short = sorted_fronts[0]
    long_id, long = sorted_fronts[-1]
    current_shared_um = float(terminal_state["enabled_max2"]["shared_process_physical_advance_um"])
    extra_below_events = sum(item["additional_events_possible_while_remaining_strictly_below_target"] for item in per_front.values())
    maximum_query_um = current_shared_um + da_um * extra_below_events
    topology_bound = {
        "schema": "pf_branching_terminal_topology_coverage_bound_v5_2_1",
        "scope": "strict bound for the exact V5.2 terminal topology only; not a universal branch-family bound",
        "terminal_step": final_step,
        "maximum_fronts": int(enabled_status["maximum_fronts"]),
        "active_front_count": len(active),
        "remaining_branch_capacity": int(enabled_status["maximum_fronts"]) - len(active),
        "fixed_topology_event_length_um": da_um,
        "shared_process_extension_um": current_shared_um,
        "total_network_new_geometry_um": float(terminal_state["enabled_max2"]["network_new_geometry_um"]),
        "maximum_forward_reach_um": float(terminal_state["enabled_max2"]["projected_extension_um"]),
        "these_are_distinct_coordinates": True,
        "target_maximum_forward_reach_um": target_um,
        "active_fronts": per_front,
        "long_arm_id": long_id,
        "short_arm_id": short_id,
        "one_below_target_long_arm_event_remains_possible": long["additional_events_possible_while_remaining_strictly_below_target"] == 1,
        "short_arm_below_target_events_remaining": short["additional_events_possible_while_remaining_strictly_below_target"],
        "total_additional_events_possible_while_all_tips_remain_below_target": extra_below_events,
        "maximum_pre_event_shared_extension_query_um": maximum_query_um,
        "one_topology_quantum_guard_um": da_um,
        "measured_endpoint_with_one_quantum_guard_um": maximum_query_um + da_um,
        "derivation": "current shared extension + fixed event length * sum of per-active-front events that can occur while that front remains strictly below target",
        "no_new_kernel_calculation_performed": True,
    }
    if not math.isclose(maximum_query_um, 740.0, abs_tol=1e-9) or not math.isclose(maximum_query_um + da_um, 745.0, abs_tol=1e-9):
        raise RuntimeError("terminal topology bound does not reproduce 740/745 micrometres")
    dump(out / "pf_branching_terminal_topology_coverage_bound.json", topology_bound)

    criteria_lines = "\n".join(
        f"{item['criterion']}. **{item['name']} — PASS.** `{json.dumps(item['evidence'], sort_keys=True, separators=(',', ':'))}`"
        for item in criteria
    )
    report = f"""# PF current-source branching corrected V5.2.1 decision

Permanent interpretation boundary: `{LABEL}`

This is an analysis-only record. It supersedes only the morphology interpretation in the immutable V5.2 decision; it does not modify the V5.2 raw results, terminal failure, mechanics, parameters, or provenance.

## Superseding scientific decision

```text
pair_terminal_result:
CORRECTED_THETA40_REPLAY_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE

atomic_topology_transaction_capability:
DEMONSTRATED

corrected_process_state_branch_birth:
DEMONSTRATED

corrected_morphology_capability:
CORRECTED_CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED_BEFORE_ENVELOPE_STOP

bounded_300um_pair_completion:
NOT_COMPLETED_SIGNED_KERNEL_ENVELOPE

final_independent_tip_mechanics:
UNQUALIFIED_ONE_OF_TWO_DAUGHTER_LOCAL_J_CONTOURS_VALID_AT_STOP

cluster_handoff_status:
UNRESOLVED_HANDOFF_NOT_REQUIRED_AT_STOP

predictive_branching_physics_validated:
false
```

The terminal result and morphology result answer different questions. The former records that neither bounded trajectory completed the 300 micrometre stopping target. The latter records what the corrected source demonstrated before the independent input-coverage stop. Three hundred micrometres is a stopping target and hard-negative ceiling, not a twelfth morphology-success gate.

## Eleven authoritative morphology criteria

{criteria_lines}

All eleven criteria pass. In addition, scalar K and tensor data share the same physical tip in all {ownership['accepted_steps_audited_all_cases']} audited accepted steps, including all {ownership['enabled_steps_at_or_after_branch_birth']} postbirth steps. The corrected process-state branch birth is therefore demonstrated, not merely reproduced provisionally.

## Coverage-planning diagnosis

The old planner reproduced

`300/cos(40 deg) + {old_margin_um:.15g} = {old_required_um:.13f} micrometres`,

then selected a 415 micrometre measured endpoint. That bound assumed one fixed +40 degree path and the inherited threshold-scaled maximum event reward. The corrected topology instead owns fixed 5 micrometre accepted arm geometry and permits +40 and -50 degree candidates.

The control accepted {len(accepted_control)} events: {control_counts[candidate_ids[1]]} at +40 degrees and {control_counts[candidate_ids[0]]} at -50 degrees. It accumulated {float(terminal_state['control_max1']['shared_process_physical_advance_um']):.0f} micrometres shared process advance while reaching only {control_projected_um:.6f} micrometres projected extension. Either next candidate would cross 300 micrometres, but selection required evaluating the 420 micrometre pre-event state, five micrometres beyond the family endpoint.

For a single front, conservative coverage must use the smallest positive projection in the actual candidate inventory. For unresolved multiple fronts it must separately track shared process extension, total network geometry, and maximum forward reach.

At the exact enabled terminal topology, the long arm can take one more event while remaining below target and the short arm can take 63. Therefore the latest possible pre-event shared-extension query is 740 micrometres. A 745 micrometre measured endpoint adds one fixed topology quantum. This is a bound only for this frozen two-front topology with no remaining branch capacity; it is not universal.

## Disposition

The corrected current-source branching morphology capability is complete for scientific purposes. The 300 micrometre pair remains formally incomplete. No PF or FEM calculation was run for V5.2.1. Extending the measured family is new deterministic FEM qualification and remains authorization-gated.

Provenance: predecessor result `{PREDECESSOR_COMMIT}`; execution source `{EXECUTION_COMMIT}` / tree `{EXECUTION_TREE}`; raw tree `{RAW_TREE_SHA256}`.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_CORRECTED_V5_2_1_DECISION.md").write_text(report)

    print(json.dumps({
        "output": str(out),
        "all_eleven_criteria_pass": all_criteria_pass,
        "corrected_morphology_capability": decision["corrected_morphology_capability"],
        "maximum_pre_event_query_um": maximum_query_um,
        "guard_endpoint_um": maximum_query_um + da_um,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
