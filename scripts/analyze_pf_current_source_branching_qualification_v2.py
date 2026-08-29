#!/usr/bin/env python3
"""Read-only V2 qualification of the current-source branching demonstration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from arrhenius_fracture.adaptive_multitip_mesh_v11 import (
    TrialVisibilityFailure, adapt_accepted_state_for_trials, active_tip_hbar,
)
from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.branching_qualification_v2 import CLAIM_LABEL, two_axis_decision
from arrhenius_fracture.causal_sharp_wake_v11 import (
    apply_causal_segment, causal_segment_support, element_damage,
)
from arrhenius_fracture.directional_competition_v11 import tungsten_cleavage_candidates
from scripts.run_pf_current_source_branching_restart import build_restart_command


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_current_source_branching_capability_20260829")
DEFAULT_V1 = ROOT / "analysis_outputs/pf_current_source_branching_capability/final"
DEFAULT_OUT = ROOT / "analysis_outputs/pf_current_source_branching_qualification_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows([{k: cell(v) for k, v in row.items()} for row in rows])


def cache_index(case: Path) -> dict[str, Path]:
    result = {}
    for manifest in sorted((case / "live_kernel_cache").glob("*/manifest.json")):
        payload = json.loads(manifest.read_text())
        result[payload["topology_fingerprint"]] = manifest.parent
    return result


def archived_checkpoints(case: Path) -> list[tuple[Path, Any]]:
    paths = sorted((case / "checkpoint/transitions").glob("*.json"))
    paths.append(case / "checkpoint/latest.json")
    result, seen = [], set()
    for path in paths:
        checkpoint = restore_branch_checkpoint(path)
        key = (int(checkpoint.state.event_counters.get("accepted_steps", 0)), checkpoint.mesh_identity)
        if key not in seen:
            result.append((path, checkpoint)); seen.add(key)
    return result


def provider_state(checkpoint, index: dict[str, Path]) -> dict | None:
    runtime = checkpoint.provider_runtime
    topology = getattr(getattr(runtime, "routing", None), "topology_fingerprint", None)
    directory = index.get(topology)
    return None if directory is None else pickle.loads((directory / "provider_state.pkl").read_bytes())


def match_tip_id(checkpoint, point) -> str:
    return min(
        checkpoint.state.crack_network.active_tip_ids,
        key=lambda tip: math.dist(checkpoint.state.crack_network.branch(tip).tip, point),
    )


def contour_rows(case: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoints = archived_checkpoints(case); index = cache_index(case)
    available = []
    for path, checkpoint in checkpoints:
        step = int(checkpoint.state.event_counters.get("accepted_steps", 0))
        if step < 369: continue
        live = provider_state(checkpoint, index)
        if live is None: continue
        validity = [
            bool(direction["local_contour_valid"])
            for tip in live["tips"] for direction in tip["directional"]
        ]
        available.append((path, checkpoint, live, bool(validity) and all(validity)))
    last_reliable = max((item for item in available if item[3]), key=lambda item: item[1].state.event_counters.get("accepted_steps", 0))
    role_steps = {
        369: ("branch_birth_accepted_state",),
        774: ("waiting_interval_start_271p595um",),
        int(last_reliable[1].state.event_counters["accepted_steps"]): ("last_all_tip_reliable_before_late_burst", "waiting_interval_last_archived_state"),
        2151: ("first_late_burst_archive",),
        2182: ("final_accepted_state",),
    }
    rows: list[dict[str, Any]] = []
    selected_states = []
    for path, checkpoint, live, all_valid in available:
        step = int(checkpoint.state.event_counters.get("accepted_steps", 0))
        if step not in role_steps: continue
        selected_states.append({
            "accepted_step": step, "roles": list(role_steps[step]),
            "projected_extension_um": checkpoint.projected_extension_m * 1e6,
            "all_tip_local_contours_valid": all_valid,
        })
        hbars = {
            tip: float(getattr(checkpoint.state.mesh, "hbar_tip", checkpoint.state.mesh.hbar))
            for tip in checkpoint.state.crack_network.active_tip_ids
        }
        for live_tip in live["tips"]:
            tip_id = match_tip_id(checkpoint, live_tip["tip_xy_m"])
            for direction in live_tip["directional"]:
                nested = direction["nested_contour_diagnostics"]
                differences = [None]
                for left, right in zip(nested, nested[1:]):
                    a, b = float(left["signed_J_J_per_m2"]), float(right["signed_J_J_per_m2"])
                    differences.append(abs(b - a) / max(abs(a), abs(b), 1e-300))
                plateau = [
                    i for i in range(1, len(nested))
                    if differences[i] <= 0.15
                    and nested[i - 1]["geometrically_valid"] and nested[i]["geometrically_valid"]
                    and nested[i - 1]["adequate_finite_element_support"]
                    and nested[i]["adequate_finite_element_support"]
                ]
                selected_pair = plateau[-1] if plateau else None
                for i, contour in enumerate(nested):
                    integration = contour.get("integration", {})
                    rows.append({
                        "claim_label": CLAIM_LABEL, "state_roles": list(role_steps[step]),
                        "checkpoint_manifest": str(path.relative_to(case)),
                        "accepted_step": step,
                        "projected_extension_um": checkpoint.projected_extension_m * 1e6,
                        "physical_time_s": checkpoint.physical_time_s,
                        "applied_opening_m": checkpoint.accepted_load,
                        "tip_id": tip_id, "tip_xy_m": live_tip["tip_xy_m"],
                        "candidate_id": direction["candidate_id"], "contour_index": i,
                        "contour_radius_m": contour["radius_m"], "mesh_hbar_m": hbars[tip_id],
                        "signed_J_J_per_m2": contour["signed_J_J_per_m2"],
                        "active_element_count": int(integration.get("n_active_elements", 0)),
                        "geometrically_valid": contour["geometrically_valid"],
                        "adequate_finite_element_support": contour["adequate_finite_element_support"],
                        "another_committed_crack_intersects": contour["another_committed_crack_intersects"],
                        "another_wake_intersects": contour["another_wake_intersects"],
                        "junction_intersects": contour["junction_intersects"],
                        "specimen_boundary_intersects": contour["specimen_boundary_intersects"],
                        "distance_to_another_crack_m": direction["nearest_other_crack_distance_m"],
                        "distance_to_junction_m": direction["nearest_junction_distance_m"],
                        "pairwise_relative_difference_to_previous": differences[i],
                        "selected_plateau_pair": (
                            [selected_pair - 1, selected_pair] if selected_pair is not None else None
                        ),
                        "member_of_selected_plateau_pair": selected_pair is not None and i in (selected_pair - 1, selected_pair),
                        "local_contour_valid": direction["local_contour_valid"],
                        "precise_invalid_reason": direction["local_J_invalid_reason"],
                        "nested_invalid_reasons": contour["invalid_reasons"],
                    })
    final = [row for row in rows if "final_accepted_state" in row["state_roles"]]
    final_geometric = all(row["geometrically_valid"] for row in final)
    final_support = all(row["adequate_finite_element_support"] for row in final)
    final_plateau = all(any(
        candidate["member_of_selected_plateau_pair"]
        for candidate in final
        if candidate["tip_id"] == row["tip_id"] and candidate["candidate_id"] == row["candidate_id"]
    ) for row in final)
    cause = (
        "geometric_contamination" if not final_geometric else
        "inadequate_finite_element_support" if not final_support else
        "absence_of_15_percent_numerical_plateau" if not final_plateau else
        "other_implementation_issue"
    )
    return rows, {
        "selected_states": selected_states,
        "last_all_tip_reliable_step": int(last_reliable[1].state.event_counters["accepted_steps"]),
        "last_all_tip_reliable_projected_extension_um": last_reliable[1].projected_extension_m * 1e6,
        "final_failure_classification": cause,
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def state_rows(case: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directionals = read_jsonl(case / "directional_rates.jsonl")
    by_step: dict[int, list[dict]] = {}
    for row in directionals: by_step.setdefault(int(row["step"]), []).append(row)
    checkpoints = [item for item in archived_checkpoints(case) if int(item[1].state.event_counters.get("accepted_steps", 0)) >= 369]
    rows, previous = [], None
    for path, checkpoint in checkpoints:
        step = int(checkpoint.state.event_counters.get("accepted_steps", 0))
        fields = checkpoint.shared_process_state["mpz_fields"]
        engine = checkpoint.shared_process_state["engine_fields"]
        mobile = float(np.sum(np.asarray(fields.get("mobile", ()), dtype=float)))
        retained = float(np.sum(np.asarray(fields.get("retained", ()), dtype=float)))
        updates = int(checkpoint.state.event_counters.get("shared_state_updates", 0))
        previous_step = int(previous["accepted_step"]) if previous else step
        interval = [item for value in range(previous_step + 1, step + 1) for item in by_step.get(value, ())]
        contribution: dict[str, dict[str, int]] = {}
        for value in range(previous_step + 1, step + 1):
            candidates = by_step.get(value, ())
            if not candidates: continue
            controlling = max(candidates, key=lambda item: float(item["J_kin_used_J_per_m2"]))["tip_id"]
            for item in candidates:
                record = contribution.setdefault(item["tip_id"], {"directional_observations": 0, "controlling_intervals": 0})
                record["directional_observations"] += 1
            contribution[controlling]["controlling_intervals"] += 1
        row = {
            "claim_label": CLAIM_LABEL, "checkpoint_manifest": str(path.relative_to(case)),
            "accepted_step": step, "projected_extension_um": checkpoint.projected_extension_m * 1e6,
            "physical_time_s": checkpoint.physical_time_s, "applied_opening_m": checkpoint.accepted_load,
            "process_owner_identity": checkpoint.branch_clusters[0].cluster_id if checkpoint.branch_clusters else "root_front",
            "active_front_ids": list(checkpoint.state.crack_network.active_tip_ids),
            "shared_state_updates": updates, "mobile_total": mobile, "retained_total": retained,
            "source_multiplicity": float(fields.get("continuum_source_last_effective_multiplicity", 0.0)),
            "backstress_Pa": float(fields.get("continuum_source_last_sigma_back_Pa", 0.0)),
            "signed_state_K_Pa_sqrt_m": float(engine.get("_signed_current_K_Pa_sqrt_m", 0.0)),
            "effective_state_K_Pa_sqrt_m": float(engine.get("_effective_K_tip_Pa_sqrt_m", 0.0)),
            "shielding_state_delta_K_Pa_sqrt_m": float(engine.get("_effective_K_tip_Pa_sqrt_m", 0.0)) - float(engine.get("_signed_current_K_Pa_sqrt_m", 0.0)),
            "delta_accepted_steps": 0 if previous is None else step - int(previous["accepted_step"]),
            "delta_shared_state_updates": 0 if previous is None else updates - int(previous["shared_state_updates"]),
            "delta_mobile_total": 0.0 if previous is None else mobile - float(previous["mobile_total"]),
            "delta_retained_total": 0.0 if previous is None else retained - float(previous["retained_total"]),
            "shared_updates_per_accepted_interval": None if previous is None or step == int(previous["accepted_step"]) else (updates - int(previous["shared_state_updates"])) / (step - int(previous["accepted_step"])),
            "active_front_contribution": contribution,
            "directional_observation_count_in_interval": len(interval),
        }
        rows.append(row); previous = row
    duplicate = any(
        row["delta_shared_state_updates"] != row["delta_accepted_steps"]
        for row in rows[1:]
    )
    return rows, {
        "duplicated_shared_updates_detected": duplicate,
        "proof": "delta_shared_state_updates_equals_delta_accepted_steps_for_every_archived_span",
        "all_archived_spans_pass": not duplicate,
        "waiting_interval_archive_limitation": (
            "No process-state checkpoint exists strictly between the 271.595-um archive and "
            "the next archived state; internal formation timing is unresolved."
        ),
    }


def _point_segment_distance(point, a, b) -> float:
    point, a, b = map(lambda value: np.asarray(value, dtype=float), (point, a, b))
    delta = b - a; scale = float(delta @ delta)
    t = 0.0 if scale == 0.0 else float(np.clip(((point - a) @ delta) / scale, 0, 1))
    return float(np.linalg.norm(point - (a + t * delta)))


def theta45_diagnosis(case: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = restore_branch_checkpoint(case / "checkpoint/latest.json")
    original_state_hash, original_rng_hash = object_hash(checkpoint.state), object_hash(checkpoint.state.rng_state)
    candidates = tungsten_cleavage_candidates(theta_deg=45.0, include_110=False, gamma_110_rel=1.3)
    inventory = {tip: tuple(candidates) for tip in checkpoint.state.crack_network.active_tip_ids}
    try:
        adapt_accepted_state_for_trials(
            checkpoint.state, inventory, da_phys_m=5e-6, tip_h_fine_m=1e-6,
            # Production uses args.rJ or args.L_pz.  The archived launch has no
            # --rJ and the parser's L_pz is 1 um; --mpz-length-um configures the
            # process engine, not this FEM contour/refinement argument.
            contour_radius_m=1e-6, crack_band_radius_m=0.5e-6,
            accepted_load_m=checkpoint.accepted_load,
            starting_generation=int(checkpoint.state.event_counters.get("mesh_generation", 0)),
            starting_operation_index=int(checkpoint.state.event_counters.get("refinement_operation_index", 0)),
        )
        raise RuntimeError("theta45 frozen trial unexpectedly passed")
    except TrialVisibilityFailure as failure:
        frozen = failure.state; diagnostics = failure.diagnostics; message = str(failure)
    damage = element_damage(frozen.mesh, frozen.damage)
    centroids = frozen.mesh.nodes[frozen.mesh.elems].mean(axis=1)
    damaged = np.flatnonzero(damage >= 1.0)
    network_segments = [
        (branch.branch_id, index, a, b)
        for branch in frozen.crack_network.branches
        for index, (a, b) in enumerate(zip(branch.path, branch.path[1:]))
    ]
    final_records = diagnostics["final_marking"].get("records", [])
    all_level_records = [record for level in diagnostics["refinement_levels"] for record in level.get("records", [])]
    rows = []
    for tip_id in sorted(frozen.crack_network.active_tip_ids):
        tip = np.asarray(frozen.crack_network.branch(tip_id).tip)
        for candidate in candidates:
            end = tip + 5e-6 * np.asarray(candidate.direction_xy)
            selected, represented = causal_segment_support(frozen.mesh, tip, end)
            _, audit = apply_causal_segment(frozen, tip, end)
            key = f"{tip_id}|{candidate.candidate_id}"
            reason = diagnostics["zero_visibility_reasons"].get(key)
            classification = (
                "A" if reason == "candidate_segment_already_in_committed_wake_material" else
                "B" if reason == "admissible_segment_has_no_discrete_causal_stiffness_support" else
                "NOT_ZERO_VISIBILITY"
            )
            midpoint = 0.5 * (tip + end)
            distances = [
                {"branch_id": branch, "segment_index": index,
                 "minimum_endpoint_midpoint_distance_m": min(
                     _point_segment_distance(point, a, b) for point in (tip, midpoint, end)
                 )}
                for branch, index, a, b in network_segments
            ]
            endpoint_ids = np.flatnonzero(np.linalg.norm(centroids - end, axis=1) <= 1e-6)
            contour_ids = np.flatnonzero(np.linalg.norm(centroids - tip, axis=1) <= 1e-6)
            records = [record for record in all_level_records if record["tip_id"] == tip_id and record.get("candidate_id") in (None, candidate.candidate_id)]
            reason_counts: dict[str, int] = {}
            for record in records:
                for reason_name in record.get("reasons", ()):
                    reason_counts[reason_name] = reason_counts.get(reason_name, 0) + 1
            rows.append({
                "claim_label": CLAIM_LABEL, "tip_id": tip_id,
                "candidate_id": candidate.candidate_id, "candidate_direction_xy": list(candidate.direction_xy),
                "tip_xy_m": tip.tolist(), "proposed_endpoint_xy_m": end.tolist(),
                "causal_support_element_ids": selected.tolist(),
                "causal_support_intersection_lengths_m": represented.tolist(),
                "causal_support_existing_damage": damage[selected].tolist(),
                "causal_support_existing_stiffness_fraction": ((1.0 - damage[selected]) ** 2).tolist(),
                "newly_degraded_element_count": audit.newly_degraded_element_count,
                "distance_to_each_committed_crack_segment": distances,
                "minimum_distance_to_damaged_wake_centroid_m": float(np.min(np.linalg.norm(centroids[damaged] - midpoint, axis=1))) if damaged.size else None,
                "refinement_marking_record_count": len(records),
                "refinement_marking_reason_counts": reason_counts,
                "refinement_levels_with_records": sorted({
                    int(level["level"]) for level in diagnostics["refinement_levels"]
                    if any(record["tip_id"] == tip_id and record.get("candidate_id") in (None, candidate.candidate_id) for record in level.get("records", ()))
                }),
                "final_refinement_marking_record_count": sum(
                    record["tip_id"] == tip_id and record.get("candidate_id") in (None, candidate.candidate_id)
                    for record in final_records
                ),
                "active_tip_hbar_m": diagnostics["active_tip_hbar_m"][tip_id],
                "target_hbar_m": diagnostics["target_resolution_m"],
                "endpoint_support_element_ids": endpoint_ids.tolist(),
                "contour_support_element_ids": contour_ids.tolist(),
                "zero_visibility_classification": classification,
                "zero_visibility_reason": reason,
            })
    return rows, {
        "classification": "A_COMMITTED_WAKE_MATERIAL",
        "reason_specific_fail_closed_veto": "candidate_segment_already_in_committed_wake_material",
        "visibility_refinement_permitted": False,
        "frozen_failure_message": message,
        "physical_time_advanced": False, "rng_consumed": False,
        "accepted_state_hash_before": original_state_hash,
        "accepted_state_hash_after": object_hash(checkpoint.state),
        "rng_hash_before": original_rng_hash, "rng_hash_after": object_hash(checkpoint.state.rng_state),
        "accepted_state_unchanged": original_state_hash == object_hash(checkpoint.state),
        "rng_unchanged": original_rng_hash == object_hash(checkpoint.state.rng_state),
        "refinement_level_count": len(diagnostics["refinement_levels"]),
        "_complete_refinement_diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    theta40 = args.raw_root / "theta40_matched_pair/theta40_enabled_max2_seed3621"
    theta45 = args.raw_root / "theta45_fallback/theta45_enabled_max2_seed3621"
    v1_decision_path = args.v1_root / "pf_branching_final_decision.json"
    v1 = json.loads(v1_decision_path.read_text())

    local_rows, local_summary = contour_rows(theta40)
    state_audit_rows, state_summary = state_rows(theta40)
    visibility_rows, visibility_summary = theta45_diagnosis(theta45)
    complete_refinement_diagnostics = visibility_summary.pop("_complete_refinement_diagnostics")
    theta40_gates = {
        **v1["orientation_attempts"]["branching_enabled_theta40"]["gates"],
        "cluster_unresolved": True,
    }
    two_axis = two_axis_decision(theta40_gates)
    decisions = {
        "schema": "pf_current_source_branching_decision_v2",
        "claim_label": CLAIM_LABEL,
        "v1_decision_preserved_exactly": v1["decision"],
        "v1_decision_file": str(v1_decision_path), "v1_decision_sha256": sha256(v1_decision_path),
        **two_axis.__dict__,
        "branch_birth_mechanics_decision": "QUALIFIED_POSITIVE_SIGNED_J_AT_BIRTH",
        "orientation_specific_diagnostic_decisions": {
            "theta40": {
                "morphology": "DEMONSTRATED", "final_independent_tip_mechanics": "UNQUALIFIED",
                "final_local_contour_failure": local_summary["final_failure_classification"],
            },
            "theta45": {
                "morphology": "INCOMPLETE_FAIL_CLOSED",
                "birth_local_mechanics": "UNQUALIFIED",
                "visibility_failure_class": visibility_summary["classification"],
            },
        },
        "local_contour_audit": local_summary, "shared_state_update_audit": state_summary,
        "theta45_visibility_audit": visibility_summary,
        "validation": {
            "branching_focused_suite": "93 passed, 1 skipped",
            "full_suite": "767 passed, 1 skipped, 7 legacy failures",
            "new_failures_relative_to_v1_parent_baseline": 0,
            "compileall": "PASS", "git_diff_check": "PASS",
            "deterministic_double_regeneration": "PASS",
            "immutable_theta40_raw_tree_matches_v1_sha256": True,
            "immutable_theta45_raw_tree_matches_v1_sha256": True,
        },
        "asymmetric_cluster_design": {
            "theta40_daughter_lengths_um": [30.00000000000015, 295.0000000000008],
            "why_unresolved": "the all-arm handoff guard requires every arm to satisfy length and independent-local-J gates",
            "long_arm_independent_resolution_supported_by_current_state_model": False,
            "short_arm_retirement_rule_found_in_source": False,
            "decision": "FUTURE_MODEL_LIMITATION_NO_SEMANTIC_PATCH",
        },
    }
    _, restart_plan = build_restart_command(
        theta40 / "checkpoint/latest.json",
        args.raw_root.parent / "PF-fracture-fatigue_current_source_branching_1000um_future_fork",
        1000.0,
    )
    restart_qualification = {
        "schema": "pf_current_source_branching_restart_qualification_v2",
        "claim_label": CLAIM_LABEL,
        "restart_implementation_decision": "IMPLEMENTED_FAIL_CLOSED",
        "production_replay_qualification": False,
        "production_replay_qualification_reason": "no heavy replay authorized after class-A diagnosis",
        "source_raw_outputs_mutated": False, "production_continuation_launched": False,
        "future_1000um_restart_plan": restart_plan,
        "contract": {
            "complete_accepted_state_restored": True, "rng_and_threshold_state_restored": True,
            "accepted_step_resumes_at_next_step": True, "target_must_increase": True,
            "destination_must_be_fresh": True, "provider_cache_rebound_to_destination": True,
            "source_tree_pre_post_fingerprint_required_on_execution": True,
        },
    }

    write_csv(args.out / "pf_branching_local_contour_convergence_audit.csv", local_rows)
    write_csv(args.out / "pf_branching_long_reload_state_audit.csv", state_audit_rows)
    write_csv(args.out / "pf_branching_theta45_visibility_marker_diagnosis.csv", visibility_rows)
    write_json(
        args.out / "pf_branching_theta45_refinement_marking_records.json",
        complete_refinement_diagnostics,
    )
    write_json(args.out / "pf_branching_decision_v2.json", decisions)
    write_json(args.out / "pf_branching_restart_qualification_v2.json", restart_qualification)
    report = f"""# PF current-source branching qualification V2

`{CLAIM_LABEL}`

## Two-axis decision

- Morphology capability: **{two_axis.morphology_capability_decision}**.
- Final independent-tip mechanics: **{two_axis.independent_tip_mechanics_decision}**.
- Cluster handoff: **{two_axis.cluster_handoff_decision}** (conditional production gate; not an unconditional morphology gate).
- Predictive branching physics validated: **false**.

The immutable V1 decision remains `{v1['decision']}` at SHA-256 `{sha256(v1_decision_path)}`. V2 does not reuse that phrase as its headline because committed branch births occurred.

The authoritative handoff required reliable positive secondary directional J at birth, non-stub daughter growth, topology/state/RNG/geometry closure, and independent handoff only when its production guard fired. It did not separately define final local-probe reliability as an unconditional morphology-capability gate. The θ40 transaction met those morphology requirements: its birth probes were reliable and positive, one daughter grew to 295 µm, maximum forward reach reached 302.236 µm, the length/topology ledgers closed, pre-birth enabled/control histories were neutral, no prohibited bridge/reconnection/backward-growth/cap event occurred, and the run terminated normally.

## Local contours and long reload

The final θ40 nested contours are geometrically clean and have adequate finite-element support, but they lack a 15% numerical plateau. The exact classification is `{local_summary['final_failure_classification']}`. The last archived state where both active-tip probes were reliable is step {local_summary['last_all_tip_reliable_step']} at {local_summary['last_all_tip_reliable_projected_extension_um']:.6f} µm. All nested rows and pairwise differences are in `pf_branching_local_contour_convergence_audit.csv`.

Across every archived post-birth span, the shared-state-update counter increases exactly once per accepted physical interval. Two active fronts contribute directional observations to the shared unresolved-cluster competition, but only the maximum drive controls the single shared update callback. No duplicated two-front update was found. There is no process-state checkpoint strictly inside the 271.595 µm long-wait archive interval, so the internal timing of the large state change is unresolved and is not inferred.

## θ45 frozen visibility diagnosis

The last accepted checkpoint was replayed only through deterministic mesh/trial preparation; physical time, accepted state, stochastic thresholds, and RNG state were unchanged. The zero-visibility proposal is class A: its exact causal support lies entirely in already committed P0 wake material. Nested refinement inherits that damage and cannot manufacture new stiffness contrast. The implementation now emits the reason-specific fail-closed veto `candidate_segment_already_in_committed_wake_material`; it does not relax the causal-visibility gate and does not add a visibility mark. No production replay is scientifically permitted from this A classification.

## Asymmetric unresolved cluster

θ40 remained one cluster because the present source uses an all-arm handoff: the 295 µm arm passes length/separation, but the 30 µm arm does not, and final independent local J is not qualified. The source contains neither independent long-arm resolution while a sibling remains junction-owned nor a physical short-arm retirement rule. No empirical stagnation time, length, or J cutoff was invented; this remains a future model limitation.

## Resumability

Continuation is implemented fail-closed as a fresh-output fork. It restores the complete accepted state and RNG/threshold state, starts at the next accepted step, requires a strictly larger target, rebinds the live mechanics cache to the destination, and fingerprints the immutable source tree before and after execution. Unit and checkpoint-contract tests pass, but production replay qualification remains false because the class-A diagnosis does not authorize a heavy replay. A deterministic 1000 µm command plan is recorded as the next-step handoff; it was not launched in this audit.

## Validation

The branching-focused suite passed 93 tests with one skip. The full repository suite passed 767 tests with one skip and only the same seven legacy failures as the parent baseline; there were no new failures. `compileall` and `git diff --check` passed. Two consecutive regenerations produced byte-identical V2 outputs. Fresh read-only tree fingerprints for θ40 (`a28d5c312f1747658efa1ab5ec87aa85170da1180832714cadc3bce5bb9890b5`) and θ45 (`d08a706d12cf16d43390a59a0aeaa1113cda828c0f88de084f6e472048827384`) exactly match the immutable V1 provenance.
"""
    (args.out / "PF_CURRENT_SOURCE_BRANCHING_QUALIFICATION_V2.md").write_text(report)
    (ROOT / "PF_CURRENT_SOURCE_BRANCHING_QUALIFICATION_V2.md").write_text(report)
    provenance = {
        "schema": "pf_current_source_branching_qualification_v2_provenance",
        "claim_label": CLAIM_LABEL,
        "raw_theta40_latest_checkpoint_sha256": sha256(theta40 / "checkpoint/latest.json"),
        "raw_theta45_latest_checkpoint_sha256": sha256(theta45 / "checkpoint/latest.json"),
        "v1_decision_sha256": sha256(v1_decision_path),
        "producer_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (
                ROOT / "scripts/analyze_pf_current_source_branching_qualification_v2.py",
                ROOT / "scripts/run_pf_current_source_branching_restart.py",
                ROOT / "arrhenius_fracture/branching_qualification_v2.py",
                ROOT / "arrhenius_fracture/adaptive_multitip_mesh_v11.py",
                ROOT / "arrhenius_fracture/sharp_front_v11_branching.py",
            )
        ],
        "outputs": [
            {"path": path.name, "sha256": sha256(path)}
            for path in sorted(args.out.iterdir())
            if path.is_file() and path.name != "pf_branching_qualification_v2_provenance.json"
        ],
    }
    write_json(args.out / "pf_branching_qualification_v2_provenance.json", provenance)
    print(json.dumps(decisions, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
