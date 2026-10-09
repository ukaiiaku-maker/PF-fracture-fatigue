#!/usr/bin/env python3
"""Instrument every element responsible for the v11 nested-refinement veto."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from arrhenius_fracture.adaptive_multitip_mesh_v11 import (
    _mean_edge_length, active_tip_hbar, mark_multitip_trial_support,
    refine_accepted_state, trial_stiffness_visibility,
)
from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.causal_sharp_wake_v11 import causal_segment_support


def triangle_metrics(points: np.ndarray) -> dict[str, float | list[float]]:
    edges = np.array([
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[1]),
        np.linalg.norm(points[0] - points[2]),
    ])
    first = points[1] - points[0]; second = points[2] - points[0]
    area = 0.5 * abs(float(first[0] * second[1] - first[1] * second[0]))
    altitudes = 2.0 * area / np.maximum(edges, np.finfo(float).tiny)
    return {
        "edge_lengths_m": edges.tolist(),
        "minimum_altitude_m": float(np.min(altitudes)),
        "maximum_edge_m": float(np.max(edges)),
        "mean_edge_m": float(np.mean(edges)),
        "area_m2": area,
        "aspect_ratio_max_edge_over_min_altitude": float(
            np.max(edges) / max(float(np.min(altitudes)), np.finfo(float).tiny)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--da-um", type=float, default=5.0)
    parser.add_argument("--target-mean-edge-um", type=float, default=1.5)
    parser.add_argument("--contour-radius-um", type=float, default=50.0)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--longest-edge-closure", action="store_true")
    args = parser.parse_args()
    restored = restore_branch_checkpoint(args.checkpoint)
    state = restored.state
    candidates = {
        tip: tuple(restored.front_competitions[tip].candidates)
        for tip in state.crack_network.active_tip_ids
    }
    da = args.da_um * 1.0e-6
    target = args.target_mean_edge_um * 1.0e-6
    contour = args.contour_radius_um * 1.0e-6
    lineage = {index: (index,) for index in range(state.mesh.ne)}
    levels = []
    for level in range(args.levels + 1):
        mesh = state.mesh
        support = mark_multitip_trial_support(
            mesh, state.crack_network, candidates, da_phys_m=da,
            contour_radius_m=contour, crack_band_radius_m=max(
                float(getattr(mesh, "hbar_tip", 0.0) or mesh.hbar), 0.5e-6,
            ),
        )
        mean_edge = _mean_edge_length(mesh)
        marked = tuple(index for index in support if mean_edge[index] > target)
        candidate_hits: dict[int, list[dict]] = {}
        for tip_id in sorted(state.crack_network.active_tip_ids):
            tip = np.asarray(state.crack_network.branch(tip_id).tip)
            for candidate in sorted(candidates[tip_id], key=lambda item: item.candidate_id):
                end = tip + da * np.asarray(candidate.direction_xy)
                ids, lengths = causal_segment_support(mesh, tip, end)
                for element_id, length in zip(ids, lengths):
                    candidate_hits.setdefault(int(element_id), []).append({
                        "tip_id": tip_id, "candidate_id": candidate.candidate_id,
                        "positive_intersection_length_m": float(length),
                        "distance_from_candidate_endpoint_m": float(
                            np.linalg.norm(mesh.nodes[mesh.elems[element_id]].mean(axis=0) - end)
                        ),
                    })
        records = []
        for element_id in marked:
            points = mesh.nodes[mesh.elems[element_id]]
            centroid = points.mean(axis=0)
            tip_distances = {
                tip_id: float(np.linalg.norm(
                    centroid - np.asarray(state.crack_network.branch(tip_id).tip)
                ))
                for tip_id in state.crack_network.active_tip_ids
            }
            endpoint_distances = {}
            for tip_id in sorted(state.crack_network.active_tip_ids):
                tip = np.asarray(state.crack_network.branch(tip_id).tip)
                for candidate in candidates[tip_id]:
                    end = tip + da * np.asarray(candidate.direction_xy)
                    endpoint_distances[f"{tip_id}|{candidate.candidate_id}"] = float(
                        np.linalg.norm(centroid - end)
                    )
            records.append({
                "triangle_id": int(element_id),
                "parent_lineage": list(lineage[element_id]),
                **triangle_metrics(points),
                "minimum_distance_to_active_tip_m": min(tip_distances.values()),
                "nearest_active_tip_id": min(tip_distances, key=tip_distances.get),
                "candidate_segments_intersected": candidate_hits.get(element_id, []),
                "inside_current_tip_J_contour": any(
                    value <= contour for value in tip_distances.values()
                ),
                "inside_candidate_endpoint_J_contour": any(
                    value <= contour for value in endpoint_distances.values()
                ),
                "minimum_distance_to_candidate_endpoint_m": min(endpoint_distances.values()),
            })
        hits = [record for record in records if record["candidate_segments_intersected"]]
        levels.append({
            "level": level, "nodes": int(mesh.nn), "elements": int(mesh.ne),
            "active_tip_hbar_m": active_tip_hbar(state),
            "candidate_new_element_counts": trial_stiffness_visibility(
                state, candidates, da_phys_m=da, crack_band_radius_m=0.0,
            ),
            "support_element_count": len(support), "veto_element_count": len(marked),
            "candidate_intersecting_veto_count": len(hits),
            "maximum_veto_mean_edge_m": max((record["mean_edge_m"] for record in records), default=0.0),
            "maximum_veto_aspect_ratio": max((record["aspect_ratio_max_edge_over_min_altitude"] for record in records), default=0.0),
            "veto_triangles": records,
        })
        if level == args.levels or not marked:
            break
        refined, operation = refine_accepted_state(
            state, marked_parent_elements=marked,
            active_tip_ids=state.crack_network.active_tip_ids,
            generation=level + 1, operation_index=level + 1,
            longest_edge_closure=args.longest_edge_closure,
        )
        next_lineage = {}
        for parent, children in operation.parent_to_child_element_map.items():
            for child in children:
                next_lineage[int(child)] = lineage[parent] + (int(child),)
        lineage = next_lineage
        state = refined
    payload = {
        "schema": "v11.refinement-veto-diagnostic/1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "da_phys_m": da, "target_mean_edge_m": target,
        "contour_radius_m": contour, "levels": levels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "v11.refinement-veto-summary/1",
        "source": str(args.out.resolve()),
        "levels": [
            {key: value for key, value in item.items() if key != "veto_triangles"}
            for item in levels
        ],
    }
    args.out.with_name(args.out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
