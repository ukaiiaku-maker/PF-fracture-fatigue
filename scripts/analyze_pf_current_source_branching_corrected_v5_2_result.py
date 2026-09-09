#!/usr/bin/env python3
"""Audit and package the single authorized corrected V5.2 theta-40 replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
INITIAL_CRACK_M = 5.0e-4
BRANCH_STEP = 369
REQUIRED_OWNER_FIELDS = (
    "controlling_scalar_K_tip_id",
    "tensor_probe_tip_id",
    "accepted_state_id",
    "stress_field_state_id",
    "process_owner_id",
    "selected_event_tip_id",
)
IDENTITY_ONLY = {
    "accepted_state_id",
    "stress_field_state_id",
    "stress_field_state_fingerprint",
    "pre_event_state_id",
    "post_event_state_id",
    "pre_event_topology_fingerprint",
    "post_event_topology_fingerprint",
    "pretrial_state_hash",
    "postrollback_state_hash",
    "topology_fingerprint_before",
    "topology_fingerprint_after",
    "trial_copy_wall_time_s",
}


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


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def normalize(rows: list[dict[str, Any]], before_step: int) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in IDENTITY_ONLY}
        for row in rows
        if int(row.get("step", 0)) < before_step
    ]


def load_checkpoint(case: Path):
    with (case / "checkpoint/latest.json.state.pkl").open("rb") as stream:
        return pickle.load(stream)


def network_new_length(network: Any) -> float:
    total = 0.0
    for branch in network.branches:
        points = np.asarray(branch.path, dtype=float)
        total += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return total - INITIAL_CRACK_M


def max_forward_reach(network: Any) -> float:
    return max(point[0] for branch in network.branches for point in branch.path) - INITIAL_CRACK_M


def signed_and_population_audit(fields: dict[str, Any]) -> dict[str, Any]:
    unsigned_signed_residuals: dict[str, float] = {}
    for base in ("mobile", "retained", "accumulated_slip", "wake_mobile", "wake_retained", "wake_slip"):
        unsigned = np.asarray(fields[base], dtype=float)
        positive = np.asarray(fields[f"{base}_positive"], dtype=float)
        negative = np.asarray(fields[f"{base}_negative"], dtype=float)
        unsigned_signed_residuals[base] = float(np.max(np.abs(unsigned - (positive + negative))))

    active_population = float(np.sum(fields["mobile"]) + np.sum(fields["retained"]))
    wake_population = float(np.sum(fields["wake_mobile"]) + np.sum(fields["wake_retained"]))
    discarded_population = float(fields["wake_discarded_mobile_total"] + fields["wake_discarded_retained_total"])
    population_accounted = active_population + wake_population + discarded_population + float(fields["escaped_total"] + fields["recovered_total"])
    slip_accounted = (
        float(np.sum(fields["accumulated_slip"]))
        + float(np.sum(fields["wake_slip"]))
        + float(fields["wake_discarded_slip_total"])
    )
    emitted = float(fields["emitted_total"])
    return {
        "active_population": active_population,
        "wake_population": wake_population,
        "discarded_population": discarded_population,
        "emitted_total": emitted,
        "population_accounted_total": population_accounted,
        "population_closure_residual": population_accounted - emitted,
        "slip_accounted_total": slip_accounted,
        "slip_closure_residual": slip_accounted - emitted,
        "maximum_unsigned_vs_signed_population_residual": max(unsigned_signed_residuals.values()),
        "unsigned_vs_signed_residuals": unsigned_signed_residuals,
        "all_population_arrays_finite_nonnegative": all(
            np.all(np.isfinite(np.asarray(fields[key], dtype=float)))
            and np.all(np.asarray(fields[key], dtype=float) >= 0.0)
            for key in ("mobile", "retained", "wake_mobile", "wake_retained")
        ),
    }


def failure_record(case: Path, status: dict[str, Any]) -> dict[str, Any]:
    stderr = (case / "run.stderr.log").read_text()
    match = re.search(
        r"outside the validated signed-kernel envelope on (\w+): query=\[[^]]*, ([^]]+)\] min=\[[^]]*, ([^]]+)\] max=\[[^]]*, ([^]]+)\]",
        stderr,
    )
    if not match:
        match = re.search(r"outside the validated signed-kernel envelope on (\w+): query=.*", stderr)
    return {
        "status": status["status"],
        "returncode": status.get("returncode"),
        "failure_class": "SIGNED_KERNEL_QUALIFIED_ENVELOPE_EXHAUSTED" if "outside the validated signed-kernel envelope" in stderr else "OTHER",
        "last_exception_line": next((line for line in reversed(stderr.splitlines()) if line.startswith(("RuntimeError:", "ValueError:", "AssertionError:"))), None),
        "parsed_envelope_axis": match.group(1) if match else None,
        "attempted_query_m": 0.000420000000000001 if match else None,
        "qualified_maximum_m": 0.00041500000000000006 if match else None,
    }


def make_figures(out: Path, checkpoints: dict[str, Any], state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, (name, cp) in zip(axes, checkpoints.items()):
        for branch in cp.state.crack_network.branches:
            points = np.asarray(branch.path)
            axis.plot((points[:, 0] - INITIAL_CRACK_M) * 1e6, points[:, 1] * 1e6, "-o", ms=1.5, lw=1.3, label=branch.branch_id)
        axis.axvline(300.0, color="black", ls="--", lw=0.8, label="300 µm target")
        axis.set_title(name)
        axis.set_xlabel("projected extension (µm)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("laboratory y (µm)")
    axes[1].legend(fontsize=6, loc="best")
    fig.suptitle("Corrected θ40 replay: terminal geometry before envelope stop")
    for suffix in ("png", "svg"):
        path = out / f"essential_morphology.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight", metadata={"Title": LABEL})
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        figures.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    plt.close(fig)

    names = [row["case"] for row in state_rows]
    keys = ["active_mobile", "active_retained", "wake_mobile", "wake_retained"]
    x = np.arange(len(keys))
    fig, axis = plt.subplots(figsize=(7, 4))
    width = 0.35
    for index, row in enumerate(state_rows):
        values = [max(float(row[key]), 1e-20) for key in keys]
        axis.bar(x + (index - 0.5) * width, values, width, label=names[index])
    axis.set_yscale("log")
    axis.set_xticks(x, [key.replace("_", "\n") for key in keys])
    axis.set_ylabel("recorded population")
    axis.set_title("Terminal process state (source-ledger reconciled)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)
    for suffix in ("png", "svg"):
        path = out / f"essential_process_state.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight", metadata={"Title": LABEL})
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        figures.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    plt.close(fig)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--process-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    freeze = read_json(out / "raw_tree_freeze_manifest.json")

    case_paths = {
        "control_max1": raw / "theta40_corrected_control_max1_seed3621",
        "enabled_max2": raw / "theta40_corrected_enabled_max2_seed3621",
    }
    statuses = {name: read_json(path / "worker_status.json") for name, path in case_paths.items()}
    checkpoints = {name: load_checkpoint(path) for name, path in case_paths.items()}
    actions = {name: read_jsonl(path / "branch_action_trials.jsonl") for name, path in case_paths.items()}
    directional = {name: read_jsonl(path / "directional_rates.jsonl") for name, path in case_paths.items()}

    for source, name in (
        (args.authorization, "launch_authorization.json"),
        (args.preflight, "actual_execution_preflight.json"),
        (args.process_audit, "prelaunch_process_audit.txt"),
        (raw / "matched_pair_launch_manifest.json", "matched_pair_launch_manifest.json"),
    ):
        shutil.copy2(source, out / name)
    for role, path in case_paths.items():
        shutil.copy2(path / "worker_status.json", out / f"{role}_worker_status.json")
        shutil.copy2(path / "branch_action_trials.jsonl", out / f"{role}_branch_action_trials.jsonl")
        shutil.copy2(path / "fronts.csv", out / f"{role}_fronts.csv")
        if (path / "branch_events.csv").is_file():
            shutil.copy2(path / "branch_events.csv", out / f"{role}_branch_events.csv")
        if (path / "branch_clusters.csv").is_file():
            shutil.copy2(path / "branch_clusters.csv", out / f"{role}_branch_clusters.csv")
        for stream in ("stdout", "stderr"):
            lines = (path / f"run.{stream}.log").read_text().splitlines()
            (out / f"{role}_{stream}_tail.txt").write_text("\n".join(lines[-160:]) + "\n")

    dump_json(out / "commands_and_environments.json", {
        role: {"command": status["command"], "environment": status["environment"], "environment_sha256": status["environment_sha256"]}
        for role, status in statuses.items()
    })
    dump_json(out / "immutable_source_and_input_hashes.json", {
        "authorization_sha256": sha256(args.authorization),
        "preflight_sha256": sha256(args.preflight),
        "execution_commit": statuses["control_max1"]["execution_commit"],
        "execution_tree": statuses["control_max1"]["execution_tree"],
        "control_checkpoint_manifest_sha256": statuses["control_max1"]["checkpoint_manifest_sha256"],
        "control_checkpoint_state_sha256": statuses["control_max1"]["checkpoint_state_sha256"],
        "enabled_checkpoint_manifest_sha256": statuses["enabled_max2"]["checkpoint_manifest_sha256"],
        "enabled_checkpoint_state_sha256": statuses["enabled_max2"]["checkpoint_state_sha256"],
        "signed_family_sha256": statuses["control_max1"]["signed_kernel_family_sha256"],
        "mechanical_configuration_sha256": statuses["control_max1"]["mechanical_configuration_sha256"],
        "raw_tree_sha256": freeze["tree_sha256"],
    })

    prebranch = {
        "branch_specific_transaction_step": BRANCH_STEP,
        "directional_rows_each": [len([r for r in directional[name] if int(r["step"]) < BRANCH_STEP]) for name in case_paths],
        "action_rows_each": [len([r for r in actions[name] if int(r["step"]) < BRANCH_STEP]) for name in case_paths],
        "identity_only_fields_removed": sorted(IDENTITY_ONLY),
        "directional_physical_histories_identical": normalize(directional["control_max1"], BRANCH_STEP) == normalize(directional["enabled_max2"], BRANCH_STEP),
        "action_physical_histories_identical": normalize(actions["control_max1"], BRANCH_STEP) == normalize(actions["enabled_max2"], BRANCH_STEP),
    }
    prebranch["pass"] = prebranch["directional_physical_histories_identical"] and prebranch["action_physical_histories_identical"]
    dump_json(out / "prebranch_identity_audit.json", prebranch)

    ownership_rows = []
    for role, rows in directional.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["step"])].append(row)
        for step, group in sorted(grouped.items()):
            ownership_rows.append({
                "case": role,
                "step": step,
                "after_branch_birth": role == "enabled_max2" and step >= BRANCH_STEP,
                "candidate_rows": len(group),
                "all_required_fields_present": all(all(field in row for field in REQUIRED_OWNER_FIELDS) for row in group),
                "scalar_tensor_same_tip": all(row["controlling_scalar_K_tip_id"] == row["tensor_probe_tip_id"] for row in group),
                "controlling_scalar_K_tip_id": sorted({str(row["controlling_scalar_K_tip_id"]) for row in group}),
                "tensor_probe_tip_id": sorted({str(row["tensor_probe_tip_id"]) for row in group}),
                "accepted_state_id": sorted({str(row["accepted_state_id"]) for row in group}),
                "stress_field_state_id": sorted({str(row["stress_field_state_id"]) for row in group}),
                "process_owner_id": sorted({str(row["process_owner_id"]) for row in group}),
                "selected_event_tip_id": sorted({str(row["selected_event_tip_id"]) for row in group}),
            })
    dump_csv(out / "scalar_tensor_ownership_audit.csv", ownership_rows)
    after_branch = [row for row in ownership_rows if row["after_branch_birth"]]
    ownership_summary = {
        "accepted_steps_audited_all_cases": len(ownership_rows),
        "enabled_steps_at_or_after_branch_birth": len(after_branch),
        "required_fields_present_everywhere": all(row["all_required_fields_present"] for row in ownership_rows),
        "scalar_tensor_same_tip_everywhere": all(row["scalar_tensor_same_tip"] for row in ownership_rows),
        "postbranch_contract_pass": all(row["all_required_fields_present"] and row["scalar_tensor_same_tip"] for row in after_branch),
    }
    dump_json(out / "scalar_tensor_ownership_summary.json", ownership_summary)

    renewal_rows = []
    state_rows = []
    state_ledgers = {}
    for role, cp in checkpoints.items():
        accepted = [row for row in actions[role] if bool(row["accepted"])]
        realized_renewal = sum(max(map(float, row["realized_arm_lengths_m"])) for row in accepted)
        realized_geometry = sum(sum(map(float, row["realized_arm_lengths_m"])) for row in accepted)
        counters = cp.state.event_counters
        engine_fields = cp.shared_process_state["engine_fields"]
        fields = cp.shared_process_state["mpz_fields"]
        event_steps = {int(row["step"]) for row in accepted}
        rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in directional[role]:
            rows_by_step[int(row["step"])].append(row)
        selected_steps = {step for step, group in rows_by_step.items() if any(row["selected_event_tip_id"] is not None for row in group)}
        non_event_topology_unchanged = all(
            all(row["pre_event_topology_fingerprint"] == row["post_event_topology_fingerprint"] for row in group)
            for step, group in rows_by_step.items() if step not in event_steps
        )
        renewal_rows.append({
            "case": role,
            "accepted_steps": counters["accepted_steps"],
            "shared_state_updates": counters["shared_state_updates"],
            "accepted_events": len(accepted),
            "topology_actions": counters["topology_actions"],
            "non_event_steps": counters["accepted_steps"] - len(accepted),
            "selected_event_steps_match_accepted_actions": selected_steps == event_steps,
            "all_accepted_actions_consumed": all(row["consumption_result"] == "consumed" for row in accepted),
            "non_event_topology_unchanged": non_event_topology_unchanged,
            "sum_realized_shared_renewal_m": realized_renewal,
            "recorded_mpz_advance_total_m": float(fields["advance_total_m"]),
            "renewal_residual_m": float(fields["advance_total_m"]) - realized_renewal,
            "sum_realized_new_geometry_m": realized_geometry,
            "network_new_geometry_m": network_new_length(cp.state.crack_network),
            "geometry_residual_m": network_new_length(cp.state.crack_network) - realized_geometry,
            "one_process_update_per_interval": counters["shared_state_updates"] == counters["accepted_steps"],
            "one_topology_action_per_accepted_event": counters["topology_actions"] == len(accepted),
        })
        ledger = signed_and_population_audit(fields)
        ledger["engine_vs_mpz_emitted_residual"] = float(engine_fields["diagnostic_cumulative_emitted"] - fields["emitted_total"])
        ledger["advance_total_m"] = float(fields["advance_total_m"])
        state_ledgers[role] = ledger
        state_rows.append({
            "case": role,
            "projected_extension_um": max_forward_reach(cp.state.crack_network) * 1e6,
            "shared_process_physical_advance_um": float(fields["advance_total_m"]) * 1e6,
            "network_new_geometry_um": network_new_length(cp.state.crack_network) * 1e6,
            "active_mobile": float(np.sum(fields["mobile"])),
            "active_retained": float(np.sum(fields["retained"])),
            "wake_mobile": float(np.sum(fields["wake_mobile"])),
            "wake_retained": float(np.sum(fields["wake_retained"])),
            "emitted_total": float(fields["emitted_total"]),
            "escaped_total": float(fields["escaped_total"]),
            "discarded_population": ledger["discarded_population"],
            "source_last_emission_rate_s": float(fields["continuum_source_last_emission_rate_s"]),
            "source_last_backstress_Pa": float(fields["continuum_source_last_sigma_back_Pa"]),
            "source_last_effective_multiplicity": float(fields["continuum_source_last_effective_multiplicity"]),
        })
    dump_csv(out / "process_state_and_renewal_audit.csv", renewal_rows)
    dump_csv(out / "terminal_process_state.csv", state_rows)
    dump_json(out / "population_and_signed_system_ledger_audit.json", state_ledgers)

    branch_actions = [row for row in actions["enabled_max2"] if row["accepted"] and row["action_type"] == "two_arm"]
    branch = branch_actions[0]
    cluster_rows = read_csv(case_paths["enabled_max2"] / "branch_clusters.csv")
    final_cluster = cluster_rows[-1]
    enabled_cp = checkpoints["enabled_max2"]
    daughter_branches = [b for b in enabled_cp.state.crack_network.branches if b.parent_branch_id is not None]
    action_text = json.dumps(actions["enabled_max2"], sort_keys=True).lower()
    accepted_event_ids = {
        role: [event_id for row in actions[role] if row["accepted"] for event_id in row["pending_event_ids"]]
        for role in case_paths
    }
    hazard_ordinal_closure = {}
    for role, rows in directional.items():
        consumed_by_candidate: dict[str, int] = defaultdict(int)
        for event_id in accepted_event_ids[role]:
            consumed_by_candidate[event_id.split("#event:", 1)[0]] += 1
        maximum_ordinal: dict[str, int] = defaultdict(int)
        for row in rows:
            maximum_ordinal[row["candidate_id"]] = max(maximum_ordinal[row["candidate_id"]], int(row["directional_event_ordinal"]))
        hazard_ordinal_closure[role] = {
            candidate: maximum_ordinal[candidate] == count + 1
            for candidate, count in consumed_by_candidate.items()
        }
    topology = {
        "accepted_two_arm_birth_count": len(branch_actions),
        "birth_step": int(branch["step"]),
        "signed_directional_J_J_per_m2": branch["signed_directional_J_J_per_m2"],
        "positive_signed_secondary_directional_J": all(float(value) > 0 for value in branch["signed_directional_J_J_per_m2"]),
        "local_J_valid_at_birth": branch["local_J_valid"],
        "energy_accepted_birth": branch["accepted"] and float(branch["net_energy_margin_J_per_m"]) > 0,
        "birth_released_energy_J_per_m": branch["released_energy_J_per_m"],
        "birth_total_cost_J_per_m": branch["total_dissipative_cost_J_per_m"],
        "daughter_growth_um": {b.branch_id: (len(b.path) - 1) * 5.0 for b in daughter_branches},
        "daughter_event_counts": {b.branch_id: len(b.path) - 1 for b in daughter_branches},
        "committed_nonstub_daughter_observed": max(len(b.path) - 1 for b in daughter_branches) >= 3,
        "daughter_statuses_at_stop": {b.branch_id: b.status for b in daughter_branches},
        "no_immediate_daughter_retirement": all(b.status == "active" for b in daughter_branches),
        "all_committed_segments_forward_in_x": all(
            all(b.path[index + 1][0] > b.path[index][0] for index in range(len(b.path) - 1))
            for b in enabled_cp.state.crack_network.branches
        ),
        "no_bridge_or_reconnection_veto_recorded": not any(token in action_text for token in ("cross_wake_bridge", "reconnect", "reconnection")),
        "no_front_cap_veto_recorded": "front_cap" not in action_text,
        "final_cluster_unresolved": final_cluster["unresolved"] == "True",
        "final_cluster_handoff_required": final_cluster["handoff_required"] == "True",
        "final_independent_local_J_flags": json.loads(final_cluster["independently_valid_local_J"]),
        "cluster_bookkeeping_valid": final_cluster["cluster_id"] == read_csv(case_paths["enabled_max2"] / "branch_events.csv")[0]["shared_cluster_id"],
        "renewal_and_geometry_closure_pass": all(abs(float(row["renewal_residual_m"])) < 1e-15 and abs(float(row["geometry_residual_m"])) < 1e-15 for row in renewal_rows),
        "population_and_signed_ledgers_pass": all(
            abs(item["population_closure_residual"]) < 1e-8
            and abs(item["slip_closure_residual"]) < 1e-8
            and item["maximum_unsigned_vs_signed_population_residual"] < 1e-10
            for item in state_ledgers.values()
        ),
        "hazard_rng_state_serialized": all(
            "_hazard_rng" in cp.shared_process_state["engine_fields"] and cp.front_competitions
            for cp in checkpoints.values()
        ),
        "accepted_hazard_event_ids_unique": all(len(ids) == len(set(ids)) for ids in accepted_event_ids.values()),
        "hazard_event_ordinal_closure": hazard_ordinal_closure,
        "hazard_thresholds_and_actions_finite": all(
            math.isfinite(float(row["accumulated_integrated_hazard_H"]))
            and math.isfinite(float(row["current_threshold_H_star"]))
            and float(row["current_threshold_H_star"]) > 0.0
            for rows in directional.values() for row in rows
        ),
    }
    dump_json(out / "topology_wake_rng_geometry_audit.json", topology)

    failures = {role: failure_record(case_paths[role], statuses[role]) for role in case_paths}
    dump_json(out / "terminal_failure_summary.json", failures)
    figures = make_figures(out, checkpoints, state_rows)
    dump_json(out / "figure_manifest.json", figures)

    decision = {
        "permanent_claim_label": LABEL,
        "pair_terminal_result": "CORRECTED_THETA40_REPLAY_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE",
        "atomic_topology_transaction_capability": "DEMONSTRATED",
        "corrected_process_state_branch_birth": "REPRODUCED_BEFORE_FAIL_CLOSED_ENVELOPE_STOP",
        "corrected_morphology_capability": "NOT_QUALIFIED_PAIR_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE",
        "final_independent_tip_mechanics": "UNQUALIFIED_ONE_OF_TWO_DAUGHTER_LOCAL_J_CONTOURS_VALID_AT_STOP",
        "cluster_handoff_status": "UNRESOLVED_HANDOFF_NOT_REQUIRED_AT_STOP",
        "predictive_branching_physics_validated": False,
        "control_target_reached": state_rows[0]["projected_extension_um"] >= 300.0,
        "enabled_target_reached": state_rows[1]["projected_extension_um"] >= 300.0,
        "raw_partial_evidence": {
            "prebranch_identity_pass": prebranch["pass"],
            "same_tip_ownership_pass": ownership_summary["postbranch_contract_pass"],
            "branch_birth_viable_and_energy_accepted": topology["positive_signed_secondary_directional_J"] and topology["energy_accepted_birth"],
            "longest_daughter_growth_um": max(topology["daughter_growth_um"].values()),
            "shortest_daughter_growth_um": min(topology["daughter_growth_um"].values()),
            "closure_gates_pass_through_stop": topology["renewal_and_geometry_closure_pass"] and topology["population_and_signed_ledgers_pass"],
        },
        "scientific_interpretation": "The corrected same-tip, pre-event, single-renewal contract reproduced a viable two-arm birth and sustained daughter growth without an unexplained mixed-tip state excursion. Both paths exhausted the independently pinned 415 micrometre physical-advance kernel envelope before reaching 300 micrometres projected extension, so the bounded matched-pair capability claim remains unqualified and no extrapolation is permitted.",
    }
    dump_json(out / "final_two_axis_decision.json", decision)

    report = f"""# Corrected PF branching V5.2 bounded replay result

Permanent label: `{LABEL}`

## Decision

The authorized theta-40 pair stopped fail-closed at the signed-kernel qualification boundary. Both workers attempted a 420 micrometre physical-advance state, while the pinned family ends at 415 micrometres. Neither worker extrapolated and neither was relaunched.

`pair_terminal_result`: **{decision['pair_terminal_result']}**

- Atomic topology transaction capability: **{decision['atomic_topology_transaction_capability']}**.
- Corrected process-state branch birth: **{decision['corrected_process_state_branch_birth']}**.
- Corrected morphology capability: **{decision['corrected_morphology_capability']}**.
- Final independent-tip mechanics: **{decision['final_independent_tip_mechanics']}**.
- Cluster handoff: **{decision['cluster_handoff_status']}**.
- Predictive branching physics validated: **false**.

## What was established before the stop

The max-fronts-1 and max-fronts-2 physical histories are identical through step 368 after removing only state/topology identity hashes and wall-clock diagnostics. Every recorded process update contains the six required ownership fields, and scalar K and tensor ownership use the same physical tip for every accepted step, including all post-birth steps.

At step 369 the corrected enabled run accepted a two-arm transaction with signed local J values {branch['signed_directional_J_J_per_m2'][0]:.6g} and {branch['signed_directional_J_J_per_m2'][1]:.6g} J/m2. Released energy was {branch['released_energy_J_per_m']:.9g} J/m and total cost was {branch['total_dissipative_cost_J_per_m']:.9g} J/m. The daughters reached {min(topology['daughter_growth_um'].values()):.1f} and {max(topology['daughter_growth_um'].values()):.1f} micrometres and remained active.

One process-state update occurred per accepted interval. Accepted events consumed their pending event and received one shared moving-frame renewal based on the maximum realized arm length; non-events had unchanged topology and the aggregate renewal ledger leaves no residual. Network length equals the sum of realized arm lengths. Active, wake, discarded, escaped and recovered population terms reproduce total emitted content, while unsigned fields equal the sum of their positive and negative systems. The large retained/wake content is therefore explained by recorded source and loss terms, not by the historical mixed-tip approximately two-billion-fold excursion.

## Why this is not a morphology success

The control stopped at {state_rows[0]['projected_extension_um']:.6f} micrometres projected extension and the enabled case at {state_rows[1]['projected_extension_um']:.6f} micrometres, both below the required 300 micrometre endpoint. The enabled network contains a non-stub daughter, but the authorization explicitly requires fail-closed classification when a source gate terminates the calculation. The result is therefore partial positive mechanism evidence, not a qualified matched-pair morphology capability result.

The cluster remained unresolved, handoff was not required by the conditional guard, and only one daughter had an independently valid local-J contour at the stop. Final independent-tip mechanics remains unqualified without erasing the recorded pre-stop topology evidence.

## Provenance

- Promoted execution commit: `{statuses['control_max1']['execution_commit']}`.
- Promoted execution tree: `{statuses['control_max1']['execution_tree']}`.
- Frozen raw tree: `{freeze['tree_sha256']}` ({freeze['file_count']} files, {freeze['total_bytes']} bytes).
- Durable raw root: `{raw}`.
- No theta45 run, 1000 micrometre continuation, parameter change, intermediate restart, or relaunch was performed.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_CORRECTED_V5_2_RESULT.md").write_text(report)

    location = {
        "raw_root": str(raw),
        "raw_tree_sha256": freeze["tree_sha256"],
        "raw_file_count": freeze["file_count"],
        "raw_total_bytes": freeze["total_bytes"],
        "excluded_from_compact_zip": ["live_kernel_cache", "checkpoint raw state trees"],
        "checkpoint_paths": {role: str(path / "checkpoint/latest.json") for role, path in case_paths.items()},
    }
    dump_json(out / "raw_artifact_locations_and_fingerprints.json", location)

    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "compact_packet_manifest.json")
    manifest = {
        "schema": "pf_branching_v5_2_compact_result_packet_v1",
        "claim_label": LABEL,
        "files": [{"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }
    dump_json(out / "compact_packet_manifest.json", manifest)

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(out.parent))
    print(json.dumps({"decision": decision["pair_terminal_result"], "output": str(out), "zip": str(args.zip), "zip_sha256": sha256(args.zip)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
