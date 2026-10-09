#!/usr/bin/env python3
"""Reconstruct the preserved 75 um crack graph with causal v11 P0 support."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.causal_sharp_wake_v11 import apply_causal_segment
from arrhenius_fracture.directional_competition_v11 import tungsten_cleavage_candidates
from arrhenius_fracture.adaptive_multitip_mesh_v11 import (
    _mean_edge_length, mark_multitip_trial_support, refine_accepted_state,
)
from arrhenius_fracture.topology_transaction_v11 import TopologyArm, clip_arm_at_first_intersection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--tip-id", default="b5f2bd5610a01132")
    parser.add_argument("--theta-deg", type=float, default=45.0)
    parser.add_argument("--da-um", type=float, default=5.0)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    accepted = restore_branch_checkpoint(args.checkpoint).state
    source_mesh = accepted.mesh
    clean_mesh = replace(source_mesh, element_damage_gp=np.zeros(source_mesh.ne))
    reconstructed = replace(accepted, mesh=clean_mesh, damage=np.zeros(clean_mesh.nn))
    old_state = accepted
    candidates = tungsten_cleavage_candidates(theta_deg=args.theta_deg)
    inventory = {tip_id: tuple(candidates) for tip_id in accepted.crack_network.active_tip_ids}
    refinement_levels = 0
    while refinement_levels < 10:
        support = mark_multitip_trial_support(
            reconstructed.mesh, reconstructed.crack_network, inventory,
            da_phys_m=args.da_um * 1.0e-6, contour_radius_m=5.0e-6,
            crack_band_radius_m=0.0,
        )
        edge = _mean_edge_length(reconstructed.mesh)
        marked = tuple(index for index in support if edge[index] > 1.5e-6)
        if not marked:
            break
        reconstructed, _ = refine_accepted_state(
            reconstructed, marked_parent_elements=marked,
            active_tip_ids=reconstructed.crack_network.active_tip_ids,
            generation=refinement_levels + 1, operation_index=refinement_levels + 1,
        )
        old_state, _ = refine_accepted_state(
            old_state, marked_parent_elements=marked,
            active_tip_ids=old_state.crack_network.active_tip_ids,
            generation=refinement_levels + 1, operation_index=refinement_levels + 1,
        )
        refinement_levels += 1
    mesh = reconstructed.mesh
    old = np.asarray(old_state.mesh.element_damage_gp, dtype=float)
    edge_count = 0
    for branch in accepted.crack_network.branches:
        for start, end in zip(branch.path, branch.path[1:]):
            reconstructed, _ = apply_causal_segment(
                reconstructed, np.asarray(start), np.asarray(end)
            )
            edge_count += 1
    new = np.asarray(reconstructed.mesh.element_damage_gp, dtype=float)
    target = accepted.crack_network.branch(args.tip_id)
    tip = np.asarray(target.tip)
    candidate_audits = []
    for candidate in candidates:
        end = tip + args.da_um * 1.0e-6 * np.asarray(candidate.direction_xy)
        _, audit = apply_causal_segment(reconstructed, tip, end)
        probe = TopologyArm(
            candidate_id=candidate.candidate_id, branch_id=args.tip_id,
            start_xy_m=tuple(tip), end_xy_m=tuple(end),
            event_reward_m=args.da_um * 1.0e-6, hazard_dissipation_J_per_m=0.0,
        )
        clipped, intersection_target = clip_arm_at_first_intersection(
            accepted.crack_network, probe
        )
        candidate_audits.append({
            "candidate_id": candidate.candidate_id,
            "start_xy_m": tip.tolist(), "end_xy_m": end.tolist(),
            "newly_degraded_element_count": audit.newly_degraded_element_count,
            "selected_element_count": len(audit.selected_element_ids),
            "selected_element_bounds_m": (
                np.stack((
                    np.min(mesh.nodes[mesh.elems[list(audit.selected_element_ids)]], axis=(0, 1)),
                    np.max(mesh.nodes[mesh.elems[list(audit.selected_element_ids)]], axis=(0, 1)),
                )).tolist() if audit.selected_element_ids else None
            ),
            "newly_degraded_element_area_m2": audit.newly_degraded_element_area_m2,
            "geometric_intersection_length_represented_m": audit.geometric_intersection_length_represented_m,
            "mechanically_resolved": audit.mechanically_resolved,
            "physical_intersection_target_branch_id": intersection_target,
            "distance_to_first_physical_intersection_m": clipped.event_reward_m,
        })
    triangles = mesh.nodes[mesh.elems]
    old_killed = old >= 1.0
    new_killed = new >= 1.0
    report = {
        "schema": "v11.causal-wake-75um-regression/1",
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "crack_representation": "sharp_wake_causal_v11",
        "physical_graph_edge_count": edge_count,
        "diagnostic_pre_refinement_levels": refinement_levels,
        "diagnostic_mesh_nodes": int(mesh.nn),
        "diagnostic_mesh_elements": int(mesh.ne),
        "old_halo_support_element_count": int(np.count_nonzero(old_killed)),
        "new_causal_support_element_count": int(np.count_nonzero(new_killed)),
        "elements_killed_only_by_old_halo": int(np.count_nonzero(old_killed & ~new_killed)),
        "elements_killed_only_by_new_causal": int(np.count_nonzero(new_killed & ~old_killed)),
        "target_tip_id": args.tip_id,
        "candidate_audits": candidate_audits,
    }
    (out / "causal_wake_75um_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    only_old = triangles[old_killed & ~new_killed]
    causal = triangles[new_killed]
    if len(only_old):
        ax.add_collection(PolyCollection(only_old * 1.0e6, facecolor="#ef4444", alpha=0.35, edgecolor="none", label="old halo only"))
    if len(causal):
        ax.add_collection(PolyCollection(causal * 1.0e6, facecolor="#2563eb", alpha=0.55, edgecolor="none", label="causal P0 support"))
    for branch in accepted.crack_network.branches:
        path = np.asarray(branch.path) * 1.0e6
        ax.plot(path[:, 0], path[:, 1], color="black", linewidth=1.3)
        if branch.status == "active":
            ax.scatter(path[-1, 0], path[-1, 1], color="#16a34a", s=28, zorder=5)
    for item in candidate_audits:
        points = np.asarray((item["start_xy_m"], item["end_xy_m"])) * 1.0e6
        ax.plot(points[:, 0], points[:, 1], linestyle="--", linewidth=2.0, label=f"{item['candidate_id']}: +{item['newly_degraded_element_count']} elems")
    ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.03)
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title("75 um blocker: old halo versus sharp_wake_causal_v11")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out / "old_vs_causal_support_overlay.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
