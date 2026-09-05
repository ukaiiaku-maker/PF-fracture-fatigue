#!/usr/bin/env python3
"""Read-only V5.4.2 completion-pair terminal audit and figure publisher."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
plt.rcParams["svg.hashsalt"] = "pf-current-source-branching-v5-4-2"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.restart_family_migration_v11 import canonical_hash
from scripts.plot_pf_branching_archived_fields_v5_4_1 import (
    event_curve, load_case, plot_network, save_triplet,
)


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
EXECUTION_COMMIT = "ae9a06d8c42287428e917baef143b0c1142cefd8"
EXECUTION_TREE = "2d0edbc7c7b51d81ab5678065ca68a1cef45d134"
A0_M = 5.0e-4
CASE_NAMES = {
    "control_max1": (
        "theta40_v5_4_2_control_max1_seed3621",
        "theta40_corrected_control_max1_seed3621",
        "control max-fronts=1",
        1,
    ),
    "enabled_max2": (
        "theta40_v5_4_2_enabled_max2_seed3621",
        "theta40_corrected_enabled_max2_seed3621",
        "branching max-fronts=2",
        2,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_clean_triplet(fig, stem: Path) -> None:
    """Save deterministic figure triplets with repository-clean SVG text."""
    save_triplet(fig, stem)
    svg = stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def value_hash(value: Any) -> str:
    try:
        return canonical_hash(value)
    except (TypeError, ValueError):
        return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def state_projection(checkpoint) -> dict[str, str]:
    state = checkpoint.state
    values = {
        "mesh_nodes": state.mesh.nodes,
        "mesh_elements": state.mesh.elems,
        "boundary": state.boundary,
        "damage": state.damage,
        "displacement": state.displacement,
        "plastic_strain": state.ep_gp,
        "dislocation_density": state.rho_gp,
        "elasticity": state.elasticity_D,
        "material": state.material,
        "cohesive_network": state.cohesive_network,
        "crack_network": state.crack_network,
        "competition": state.competition,
        "tip_process_state": state.tip_process_state,
        "junction_process_state": state.junction_process_state,
        "energy_ledgers": state.energy_ledgers,
        "rng_state": state.rng_state,
        "event_counters": state.event_counters,
        "stored_energy": state.stored_energy_J_per_m,
        "physical_time": checkpoint.physical_time_s,
        "accepted_load": checkpoint.accepted_load,
        "projected_extension": checkpoint.projected_extension_m,
        "physical_extension": checkpoint.physical_extension_m,
        "front_competitions": checkpoint.front_competitions,
        "branch_clusters": checkpoint.branch_clusters,
        "handoff_guard_diagnostics": checkpoint.handoff_guard_diagnostics,
    }
    engine = {
        key: value for key, value in checkpoint.shared_process_state["engine_fields"].items()
        if key not in {
            "_state_kernel_family", "_restart_family_migration_audit",
            "_restart_family_migration_count",
        }
    }
    mpz = {
        key: value for key, value in checkpoint.shared_process_state["mpz_fields"].items()
        if key != "_signed_kernel"
    }
    values["engine_without_family_provenance"] = engine
    for key, value in sorted(mpz.items()):
        values[f"mpz.{key}"] = value
    return {key: value_hash(value) for key, value in values.items()}


def prefix_audit(new_case: Path, old_case: Path, role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exact_files = (
        "fronts.csv", "energy_ledger.csv", "provider_transitions.csv",
        "directional_rates.jsonl",
    )
    if role == "enabled_max2":
        exact_files += ("branch_events.csv", "branch_clusters.csv")
    for name in exact_files:
        old_bytes = (old_case / name).read_bytes()
        new_bytes = (new_case / name).read_bytes()
        exact = new_bytes.startswith(old_bytes)
        rows.append({
            "role": role, "kind": "byte_prefix", "item": name,
            "old_count_or_size": len(old_bytes), "new_count_or_size": len(new_bytes),
            "exact": exact,
        })
        if not exact:
            raise RuntimeError(f"V5.2 byte prefix differs for {role}/{name}")

    old_actions = read_jsonl(old_case / "branch_action_trials.jsonl")
    new_actions = read_jsonl(new_case / "branch_action_trials.jsonl")
    semantic_exact = len(new_actions) >= len(old_actions)
    for old, new in zip(old_actions, new_actions):
        old = {key: value for key, value in old.items() if key != "trial_copy_wall_time_s"}
        new = {key: value for key, value in new.items() if key != "trial_copy_wall_time_s"}
        semantic_exact = semantic_exact and old == new
    rows.append({
        "role": role, "kind": "semantic_prefix",
        "item": "branch_action_trials.jsonl excluding trial_copy_wall_time_s",
        "old_count_or_size": len(old_actions), "new_count_or_size": len(new_actions),
        "exact": semantic_exact,
    })
    if not semantic_exact:
        raise RuntimeError(f"V5.2 action semantic prefix differs for {role}")

    old_transitions = sorted((old_case / "checkpoint/transitions").glob("*.json"))
    matched_steps = []
    for old_path in old_transitions:
        new_path = new_case / "checkpoint/transitions" / old_path.name
        if not new_path.is_file():
            raise RuntimeError(f"new run lacks archived V5.2 transition {new_path}")
        old_projection = state_projection(restore_branch_checkpoint(old_path))
        new_projection = state_projection(restore_branch_checkpoint(new_path))
        exact = old_projection == new_projection
        matched_steps.append(int(old_path.name.split("_", 1)[0].replace("step", "")))
        rows.append({
            "role": role, "kind": "accepted_checkpoint_physical_stochastic_identity",
            "item": old_path.name, "old_count_or_size": len(old_projection),
            "new_count_or_size": len(new_projection), "exact": exact,
        })
        if not exact:
            differences = sorted(key for key in old_projection if old_projection[key] != new_projection.get(key))
            raise RuntimeError(f"V5.2 checkpoint prefix differs for {role}/{old_path.name}: {differences}")
    return {
        "qualification": "PASS",
        "byte_exact_files": list(exact_files),
        "action_trial_semantic_prefix_exact": True,
        "excluded_nondeterministic_action_field": "trial_copy_wall_time_s",
        "accepted_checkpoint_count": len(old_transitions),
        "last_accepted_transition_step": max(matched_steps),
        "physical_and_stochastic_state_projection_exact": True,
        "intentional_exclusions": [
            "signed-kernel family object", "restart-family migration provenance",
            "provider cache path/identity",
        ],
    }, rows


def terminal_audit(case: Path, role: str, maximum_fronts: int) -> dict[str, Any]:
    checkpoint = restore_branch_checkpoint(case / "checkpoint/latest.json")
    manifest = json.loads((case / "checkpoint/latest.json").read_text())
    complete = json.loads((case / "run_complete.json").read_text())
    actions = read_jsonl(case / "branch_action_trials.jsonl")
    accepted = [row for row in actions if row.get("accepted")]
    directionals = read_jsonl(case / "directional_rates.jsonl")
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in directionals:
        by_step.setdefault(int(row["step"]), []).append(row)

    same_tip = all(
        row["controlling_scalar_K_tip_id"] == row["tensor_probe_tip_id"]
        for row in directionals
    )
    stress_identity = all(
        row["stress_field_state_id"] == row["stress_field_state_fingerprint"]
        for row in directionals
    ) and all(
        len({row["accepted_state_id"] for row in rows}) == 1
        and len({row["stress_field_state_id"] for row in rows}) == 1
        for rows in by_step.values()
    ) and all(
        row["pretrial_state_hash"] == row["accepted_state_id"] for row in accepted
    )

    transition_manifests = sorted((case / "checkpoint/transitions").glob("*.json"))
    checkpoint_update_exact = True
    for path in transition_manifests + [case / "checkpoint/latest.json"]:
        archived = restore_branch_checkpoint(path)
        counters = archived.state.event_counters
        checkpoint_update_exact = checkpoint_update_exact and (
            int(counters.get("shared_state_updates", -1))
            == int(counters.get("accepted_steps", -2))
        )

    counters = checkpoint.state.event_counters
    topology_actions = int(counters.get("topology_actions", -1))
    accepted_steps = int(counters.get("accepted_steps", -1))
    shared_updates = int(counters.get("shared_state_updates", -1))
    accepted_action_count = len(accepted)
    accepted_arm_count = sum(len(row["candidate_ids"]) for row in accepted)
    topology_once = topology_actions == accepted_action_count
    non_event_count = accepted_steps - accepted_action_count
    all_accepted_realized = all(
        row["geometry_status"] == "realized"
        and row["reservation_result"] == "accepted"
        and all(abs(float(value) - 5e-6) <= 1e-18 for value in row["realized_arm_lengths_m"])
        for row in accepted
    )
    mpz = checkpoint.shared_process_state["mpz_fields"]
    renewal_distance = float(mpz["advance_total_m"])
    renewal_exact = abs(renewal_distance - accepted_action_count * 5e-6) <= 1e-15

    active_wake_sinks = sum(float(np.sum(np.asarray(mpz[key], dtype=float))) for key in (
        "mobile", "retained", "wake_mobile", "wake_retained",
        "wake_discarded_mobile_total", "wake_discarded_retained_total",
        "escaped_total", "recovered_total",
    ))
    emitted = float(np.sum(np.asarray(mpz["emitted_total"], dtype=float)))
    conservation_residual = active_wake_sinks - emitted
    conservation_relative = abs(conservation_residual) / max(abs(emitted), 1.0)
    signed_split_residual = 0.0
    for base in (
        "mobile", "retained", "wake_mobile", "wake_retained",
        "accumulated_slip", "wake_slip",
    ):
        residual = np.asarray(mpz[base]) - (
            np.asarray(mpz[base + "_positive"]) + np.asarray(mpz[base + "_negative"])
        )
        signed_split_residual = max(signed_split_residual, float(np.max(np.abs(residual))))

    unique_competitions = {}
    for competition in checkpoint.front_competitions.values():
        unique_competitions[value_hash(competition)] = competition
    competition_event_indices = {state.competition_event_index for state in unique_competitions.values()}
    completed_directional_events = sum(
        hazard.completed_event_count
        for state in unique_competitions.values() for hazard in state.hazard_states
    )
    pending_events = sum(
        len(hazard.pending_events)
        for state in unique_competitions.values() for hazard in state.hazard_states
    )
    thresholds_open = all(
        hazard.action < hazard.current_threshold_action
        for state in unique_competitions.values() for hazard in state.hazard_states
    )
    hazard_closure = (
        competition_event_indices == {accepted_action_count}
        and completed_directional_events == accepted_arm_count
        and pending_events == 0 and thresholds_open
    )

    network = checkpoint.state.crack_network
    branch_paths = [np.asarray(branch.path, dtype=float) for branch in network.branches]
    forward_only = all(np.all(np.diff(path[:, 0]) > 0.0) for path in branch_paths)
    active_ids = list(network.active_tip_ids)
    vetoes = [str(row.get("veto_reason") or "") for row in actions]
    cap_bound = any("capacity" in veto.lower() or "maximum_front" in veto.lower() for veto in vetoes)
    prohibited_veto = any(
        token in veto.lower() for veto in vetoes
        for token in ("bridge", "reconnect", "reversal", "backward")
    )
    if role == "enabled_max2":
        birth = read_csv(case / "branch_events.csv")[0]
        daughter_ids = json.loads(birth["arm_front_ids"])
    else:
        daughter_ids = active_ids
    no_retirement = (
        int(manifest.get("merged_front_count", 0)) == 0
        and all(daughter in active_ids for daughter in daughter_ids)
        and all(branch.status == "active" for branch in network.branches if branch.branch_id in daughter_ids)
    )

    handoff = {"guard_evaluated": role == "enabled_max2", "guard_fired": False, "passed": True}
    if role == "enabled_max2":
        cluster_rows = read_csv(case / "branch_clusters.csv")
        final_cluster = cluster_rows[-1]
        sufficient = json.loads(final_cluster["sufficient_post_junction_length"])
        independent = json.loads(final_cluster["independently_valid_local_J"])
        guard_fired = (
            all(sufficient)
            and final_cluster["separation_reaches_process_zone"] == "True"
            and final_cluster["local_contours_overlap"] == "False"
            and all(independent)
        )
        handoff = {
            "guard_evaluated": True,
            "guard_fired": guard_fired,
            "handoff_required": final_cluster["handoff_required"] == "True",
            "cluster_unresolved": final_cluster["unresolved"] == "True",
            "arm_lengths_um": [value * 1e6 for value in json.loads(final_cluster["arm_lengths_m"])],
            "passed": (not guard_fired and final_cluster["handoff_required"] == "False"),
        }

    checks = {
        "target_reached": (
            complete.get("status") == "target_reached"
            and manifest.get("termination_reason") == "target_reached"
            and checkpoint.projected_extension_m * 1e6 >= 300.0
        ),
        "same_tip_scalar_K_tensor_ownership_every_interval": same_tip,
        "accepted_pre_event_stress_state_identity": stress_identity,
        "one_process_update_per_interval": (
            checkpoint_update_exact and shared_updates == accepted_steps
        ),
        "zero_topology_renewal_for_non_events": topology_once and non_event_count >= 0,
        "one_realized_geometry_renewal_for_each_accepted_event": (
            all_accepted_realized and renewal_exact
        ),
        "active_plus_wake_plus_sinks_conservation": conservation_relative <= 1e-12,
        "signed_system_conservation": signed_split_residual == 0.0,
        "hazard_rng_closure": hazard_closure,
        "no_bridge_reconnection_reversal": forward_only and not prohibited_veto,
        "no_cap_binding": not cap_bound and len(active_ids) <= maximum_fronts,
        "no_immediate_daughter_retirement": no_retirement,
        "conditional_independent_handoff_guard": bool(handoff["passed"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"terminal audit failed for {role}: {checks}")
    return {
        "qualification": "PASS",
        "checks": checks,
        "termination_reason": manifest.get("termination_reason"),
        "accepted_steps": accepted_steps,
        "accepted_event_transactions": accepted_action_count,
        "accepted_directional_events": accepted_arm_count,
        "non_event_intervals": non_event_count,
        "topology_actions": topology_actions,
        "shared_state_updates": shared_updates,
        "maximum_network_forward_reach_um": checkpoint.projected_extension_m * 1e6,
        "accumulated_physical_path_um": checkpoint.physical_extension_m * 1e6,
        "active_front_ids": active_ids,
        "committed_branch_birth_count": int(manifest.get("committed_branch_birth_count", 0)),
        "renewal_distance_um": renewal_distance * 1e6,
        "active_wake_sinks_conservation_residual": conservation_residual,
        "active_wake_sinks_conservation_relative": conservation_relative,
        "signed_split_max_absolute_residual": signed_split_residual,
        "completed_directional_events": completed_directional_events,
        "pending_directional_events": pending_events,
        "handoff": handoff,
        "live_fem_solve_count": int(complete["validation"]["live_fem_solve_count"]),
        "accepted_provider_state_count": int(complete["validation"]["accepted_provider_state_count"]),
    }


def nodal_field(ax, checkpoint, values, title: str, cmap: str = "viridis"):
    state = checkpoint.state
    tri = mtri.Triangulation(state.mesh.nodes[:, 0] * 1e6, state.mesh.nodes[:, 1] * 1e6, state.mesh.elems)
    artist = ax.tricontourf(tri, np.asarray(values), levels=24, cmap=cmap)
    ax.set(title=title, xlabel="laboratory x (µm)", ylabel="laboratory y (µm)")
    ax.set_aspect("equal", adjustable="box")
    plot_network(ax, checkpoint)
    return artist


def element_field(ax, checkpoint, values, title: str, cmap: str = "magma"):
    state = checkpoint.state
    tri = mtri.Triangulation(state.mesh.nodes[:, 0] * 1e6, state.mesh.nodes[:, 1] * 1e6, state.mesh.elems)
    values = np.asarray(values)
    if values.ndim > 1:
        n_elements = state.mesh.elems.shape[0]
        if values.shape[0] == n_elements:
            values = np.linalg.norm(values.reshape((n_elements, -1)), axis=1)
        elif values.shape[-1] == n_elements:
            values = np.linalg.norm(np.moveaxis(values, -1, 0).reshape((n_elements, -1)), axis=1)
        else:
            raise RuntimeError(f"cannot map element field {values.shape} onto {n_elements} elements")
    artist = ax.tripcolor(tri, facecolors=values, shading="flat", cmap=cmap)
    ax.set(title=title, xlabel="laboratory x (µm)", ylabel="laboratory y (µm)")
    ax.set_aspect("equal", adjustable="box")
    plot_network(ax, checkpoint)
    return artist


def figures(raw_root: Path, out: Path) -> dict[str, Any]:
    cases = {
        label: raw_root / new_name
        for new_name, _, label, _ in CASE_NAMES.values()
    }
    loaded = {label: load_case(path) for label, path in cases.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    for ax, (label, (checkpoint, _, _)) in zip(axes, loaded.items()):
        artist = nodal_field(ax, checkpoint, checkpoint.state.damage, f"{label}\nfinal nodal damage")
        ax.set_xlim(450, 850); ax.set_ylim(-230, 230); ax.legend(fontsize=7, loc="upper left")
        fig.colorbar(artist, ax=ax, shrink=0.82, label="damage d")
    fig.suptitle("Corrected theta-40 V5.4.2 completion topology at 300 µm maximum forward reach")
    save_clean_triplet(fig, out / "pf_branching_completion_final_crack_structure_v5_4_2")
    plt.close(fig)

    checkpoint = loaded["branching max-fronts=2"][0]
    state = checkpoint.state
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.0), constrained_layout=True)
    displacement = np.asarray(state.displacement)
    if displacement.ndim == 1:
        displacement = displacement.reshape((-1, 2))
    panels = (
        (axes[0, 0], state.damage, "nodal damage d", "viridis", True),
        (axes[0, 1], np.linalg.norm(displacement, axis=1) * 1e6, "displacement magnitude (µm)", "magma", True),
        (axes[0, 2], state.ep_gp, "element plastic-strain magnitude", "plasma", False),
        (axes[1, 0], state.rho_gp, "element dislocation-density magnitude", "cividis", False),
    )
    for ax, values, title, cmap, nodal in panels:
        artist = nodal_field(ax, checkpoint, values, title, cmap) if nodal else element_field(ax, checkpoint, values, title, cmap)
        ax.set_xlim(450, 850); ax.set_ylim(-230, 230)
        fig.colorbar(artist, ax=ax, shrink=0.78)
    mpz = checkpoint.shared_process_state["mpz_fields"]
    x = np.asarray(mpz["x"]) * 1e6
    wx = np.asarray(mpz["wake_x"]) * 1e6
    for system in range(int(mpz["n_systems"])):
        axes[1, 1].plot(x, np.asarray(mpz["mobile"])[system], lw=1.6, label=f"active mobile s{system}")
        axes[1, 1].plot(x, np.asarray(mpz["retained"])[system], "--", lw=1.6, label=f"active retained s{system}")
        axes[1, 2].plot(wx, np.asarray(mpz["wake_mobile"])[system], lw=1.5, label=f"wake mobile s{system}")
        axes[1, 2].plot(wx, np.asarray(mpz["wake_retained"])[system], "--", lw=1.5, label=f"wake retained s{system}")
    axes[1, 1].set(title="tip-relative active MPZ populations", xlabel="active MPZ coordinate (µm)", ylabel="line content")
    axes[1, 2].set(title="laboratory wake populations", xlabel="wake coordinate (µm)", ylabel="line content")
    axes[1, 1].legend(fontsize=7); axes[1, 2].legend(fontsize=7)
    fig.suptitle("Branching-enabled V5.4.2 terminal FE and process-state fields")
    save_clean_triplet(fig, out / "pf_branching_completion_final_damage_and_process_fields_v5_4_2")
    plt.close(fig)

    all_rows = []
    identities = {}
    for label, case in cases.items():
        rows, mapping, birth_step = event_curve(case, label)
        all_rows.extend(rows)
        identities[label] = {"candidate_to_front_after_birth": mapping, "birth_step": birth_step}
    table = out / "pf_branching_completion_model_native_accepted_event_kinetic_K_v5_4_2.csv"
    write_csv(table, all_rows)

    colors = {"control max-fronts=1": "#4c78a8", "branching max-fronts=2": "#f58518"}
    for coordinate, xlabel, stem, title in (
        (
            "maximum_network_forward_reach_um", "maximum network forward reach (µm)",
            "pf_branching_completion_model_native_accepted_event_kinetic_K_vs_maximum_network_forward_reach_v5_4_2",
            "PF model-native accepted-event kinetic K versus maximum network forward reach",
        ),
        (
            "selected_event_front_projected_extension_um", "selected-event front projected progress (µm)",
            "pf_branching_completion_model_native_accepted_event_kinetic_K_vs_selected_front_progress_v5_4_2",
            "PF model-native accepted-event kinetic K versus selected-front progress",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9.4, 6.0), constrained_layout=True)
        for label in cases:
            subset = [row for row in all_rows if row["case"] == label]
            valid = [row for row in subset if row["local_J_valid"]]
            fallback = [row for row in subset if not row["local_J_valid"]]
            ax.scatter([row[coordinate] for row in valid], [row["kinetic_K_used_MPa_sqrt_m"] for row in valid],
                       s=20, marker="o", color=colors[label], alpha=0.78,
                       label=f"{label}: valid local contour")
            ax.scatter([row[coordinate] for row in fallback], [row["kinetic_K_used_MPa_sqrt_m"] for row in fallback],
                       s=30, marker="x", color=colors[label], alpha=0.88,
                       label=f"{label}: marginal-energy fallback")
        ax.set(xlabel=xlabel, ylabel=r"kinetic $K$ used (MPa√m)", title=title)
        ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
        ax.text(0.01, 0.01,
                "Accepted-event model-native drive only; not fracture toughness or an R-curve.\n"
                "Fallback crosses are not connected as qualified local-contour values.",
                transform=ax.transAxes, fontsize=8, va="bottom")
        save_clean_triplet(fig, out / stem)
        plt.close(fig)

    outputs = []
    for path in sorted(out.glob("pf_branching_completion_*v5_4_2.*")):
        if path.suffix in {".csv", ".png", ".pdf", ".svg"}:
            outputs.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "qualification": "PASS",
        "event_owner_source": "directional_rates.selected_event_tip_id",
        "candidate_owner_source": "branch_events.event_ids_consumed paired with arm_front_ids at birth",
        "identities": identities,
        "invalid_contour_points_connected": False,
        "full_history_called_fracture_toughness_or_R_curve": False,
        "mechanics_solve_performed": False,
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--v5-2-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    raw_root = args.raw_root.resolve(); old_root = args.v5_2_root.resolve()
    pair = json.loads(args.pair_manifest.read_text())
    if not pair.get("execution_success") or not pair.get("immutable_inputs_exact"):
        raise RuntimeError("pair launch manifest is not successful and immutable")
    if pair.get("execution_commit") != EXECUTION_COMMIT or pair.get("execution_tree") != EXECUTION_TREE:
        raise RuntimeError("pair used an unreviewed execution source")

    prefix = {}; prefix_rows = []; terminal = {}
    for role, (new_name, old_name, _, maximum) in CASE_NAMES.items():
        prefix[role], rows = prefix_audit(raw_root / new_name, old_root / old_name, role)
        prefix_rows.extend(rows)
        terminal[role] = terminal_audit(raw_root / new_name, role, maximum)
    write_csv(out / "pf_branching_completion_v5_2_prefix_parity_v5_4_2.csv", prefix_rows)
    figure_audit = figures(raw_root, out)

    all_pass = all(value["qualification"] == "PASS" for value in prefix.values()) and all(
        value["qualification"] == "PASS" for value in terminal.values()
    ) and figure_audit["qualification"] == "PASS"
    decision = {
        "schema": "pf_branching_completion_terminal_audit_v5_4_2/1",
        "qualification": "PASS" if all_pass else "FAIL_CLOSED",
        "boundary": BOUNDARY,
        "execution_commit": EXECUTION_COMMIT,
        "execution_tree": EXECUTION_TREE,
        "pair_manifest": {"path": str(args.pair_manifest.resolve()), "sha256": sha256(args.pair_manifest)},
        "raw_root": str(raw_root),
        "raw_tree_hashes_before_analysis": pair["raw_output_trees_before_analysis"],
        "prefix_parity": prefix,
        "terminal_cases": terminal,
        "figure_audit": figure_audit,
        "pair_terminal_result": "CORRECTED_THETA40_COMPLETION_PAIR_REACHED_300UM" if all_pass else "FAIL_CLOSED",
        "bounded_300um_pair_completion": "COMPLETED" if all_pass else "NOT_QUALIFIED",
        "corrected_process_state_branch_birth": "DEMONSTRATED",
        "corrected_morphology_capability": "CORRECTED_CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED",
        "maximum_fronts_greater_than_two_supported": False,
        "continuation_to_1000um_authorized": False,
        "predictive_branching_physics_validated": False,
        "analysis_pf_workers_started": 0,
        "analysis_mechanics_solves": 0,
    }
    if not all_pass:
        raise RuntimeError("V5.4.2 completion qualification failed")
    write_json(out / "pf_branching_completion_terminal_audit_v5_4_2.json", decision)

    control = terminal["control_max1"]
    enabled = terminal["enabled_max2"]
    report = f"""# PF current-source branching completion execution V5.4.2

Permanent interpretation boundary: `{BOUNDARY}`.

## Decision

- `pair_terminal_result: CORRECTED_THETA40_COMPLETION_PAIR_REACHED_300UM`
- `bounded_300um_pair_completion: COMPLETED`
- `corrected_process_state_branch_birth: DEMONSTRATED`
- `corrected_morphology_capability: CORRECTED_CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED`
- `predictive_branching_physics_validated: false`
- `continuation_to_1000um_authorized: false`
- `maximum_fronts_greater_than_two_supported: false`

The immutable execution source is `{EXECUTION_COMMIT}` with tree `{EXECUTION_TREE}`. The pair launcher used one completion worker at a time because one unrelated PF-sintering worker was active; the system-wide total never exceeded two. Both worker return codes were zero, raw trees were frozen before analysis, and every pinned input remained byte-identical.

## Terminal results

The max-fronts=1 control reached `{control['maximum_network_forward_reach_um']:.12g}` µm with one active front after `{control['accepted_steps']}` accepted intervals and `{control['accepted_event_transactions']}` topology transactions. The max-fronts=2 case reached `{enabled['maximum_network_forward_reach_um']:.12g}` µm with active fronts `{', '.join(enabled['active_front_ids'])}`, one committed branch birth, and daughter lengths `{enabled['handoff']['arm_lengths_um'][0]:.12g}` and `{enabled['handoff']['arm_lengths_um'][1]:.12g}` µm.

Every terminal gate passed: same-tip scalar-K/tensor ownership, accepted pre-event stress identity, one shared process update per interval, zero topology renewal on non-events, one realized-geometry renewal per accepted topology transaction, active-plus-wake-plus-sinks conservation, signed positive/negative decomposition, hazard/RNG closure, no bridge/reconnection/reversal, no cap binding, and no immediate daughter retirement. The independent-handoff guard did not fire because the short arm and final independent-local-J condition did not satisfy the all-arm policy; the unresolved cluster is therefore expected, not a failure.

## V5.2 prefix parity

The complete corrected V5.2 table prefix is byte-identical in both new cases for fronts, energy, provider transitions, directional histories, and all applicable branch tables. Action-trial records are semantically identical after excluding only `trial_copy_wall_time_s`. All `{prefix['control_max1']['accepted_checkpoint_count']}` control and `{prefix['enabled_max2']['accepted_checkpoint_count']}` enabled V5.2 transition checkpoints reproduce the physical, topology, process, hazard, RNG, time, and opening state exactly after excluding only the intentionally replaced signed-kernel family object, restart-family provenance, and cache identity. The step-369 branch birth is therefore exactly reproduced.

## Figures and interpretation

The completion figures are generated only from the new terminal outputs and do not overwrite V5.4.1. Kinetic-K histories are explicitly model-native accepted-event diagnostics. They are not called fracture toughness or R-curves. Valid local-contour points use circles; marginal-energy fallbacks use unconnected crosses. Separate figures use maximum network forward reach and selected-event front progress, respectively.

This optional endpoint closure strengthens the archived capability record but does not calibrate branching probability, validate recursive branching, establish `maximum_fronts > 2`, or validate predictive branching physics.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_COMPLETION_EXECUTION_V5_4_2.md").write_text(report)
    provenance = {
        "schema": "pf_branching_completion_execution_provenance_v5_4_2/1",
        "qualification": "PASS",
        "boundary": BOUNDARY,
        "producer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs": [
            {"path": str(args.pair_manifest.resolve()), "sha256": sha256(args.pair_manifest)},
            {"path": str((raw_root / "pair_manifest.json").resolve()), "sha256": sha256(raw_root / "pair_manifest.json")},
        ],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "pf_branching_completion_execution_provenance_v5_4_2.json"
        ],
        "workers_started": 0,
        "mechanics_solves": 0,
    }
    write_json(out / "pf_branching_completion_execution_provenance_v5_4_2.json", provenance)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
