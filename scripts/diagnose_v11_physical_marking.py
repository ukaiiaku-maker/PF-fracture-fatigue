#!/usr/bin/env python3
"""Reason-resolved replay of the v11 physical refinement controller."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from arrhenius_fracture.adaptive_multitip_mesh_v11 import (
    MarkingReason, active_tip_hbar, diagnose_underresolved_trial_geometry,
    refine_accepted_state, trial_stiffness_visibility,
)
from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--da-um", type=float, default=5.0)
    parser.add_argument("--target-um", type=float, default=1.5)
    parser.add_argument("--contour-um", type=float, default=1.0)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    restored = restore_branch_checkpoint(args.checkpoint)
    state = restored.state
    candidates = {
        tip: tuple(restored.front_competitions[tip].candidates)
        for tip in state.crack_network.active_tip_ids
    }
    da = args.da_um * 1e-6
    target = args.target_um * 1e-6
    contour = args.contour_um * 1e-6
    root_lineage = {element_id: element_id for element_id in range(state.mesh.ne)}
    parent_records = None
    parent_map = None
    conformity_elements: set[int] = set()
    levels = []
    all_roots: set[int] = set()
    all_leaf_ids: set[tuple[int, int]] = set()
    cumulative = 0

    for level in range(args.levels + 1):
        audit = diagnose_underresolved_trial_geometry(
            state.mesh, state.crack_network, candidates, da_phys_m=da,
            contour_radius_m=contour, target_resolution_m=target,
        )
        records = []
        reason_counts = Counter()
        association_counts = Counter()
        monotonic = []
        previous = {}
        if parent_records is not None and parent_map is not None:
            for record in parent_records.records:
                for child in parent_map.get(record.element_id, ()):
                    previous[(child, record.tip_id, record.candidate_id)] = record
        for record in audit.records:
            row = record.to_dict()
            row["parent_lineage_root_element_id"] = int(root_lineage[record.element_id])
            records.append(row)
            for reason in record.reasons:
                reason_counts[reason] += 1
                association_counts[(record.tip_id, record.candidate_id or "-", reason)] += 1
            prior = previous.get((record.element_id, record.tip_id, record.candidate_id))
            if prior is not None:
                ratio = record.controlling_metric_m / max(
                    prior.controlling_metric_m, np.finfo(float).tiny,
                )
                monotonic.append({
                    "element_id": record.element_id, "tip_id": record.tip_id,
                    "candidate_id": record.candidate_id,
                    "parent_controlling_metric_m": prior.controlling_metric_m,
                    "child_controlling_metric_m": record.controlling_metric_m,
                    "threshold_m": target, "child_over_parent_ratio": ratio,
                    "decreased": ratio < 1.0,
                })
        marked = audit.marked_element_ids
        cumulative += len(marked)
        roots = {root_lineage[element_id] for element_id in marked}
        all_roots.update(roots)
        all_leaf_ids.update((level, element_id) for element_id in marked)
        if marked:
            nodes = state.mesh.nodes[state.mesh.elems[list(marked)]].reshape(-1, 2)
            bbox = {"minimum_xy_m": nodes.min(axis=0).tolist(), "maximum_xy_m": nodes.max(axis=0).tolist()}
            area = float(np.sum(state.mesh.area_e[list(marked)]))
            eq = np.sqrt(4.0 * state.mesh.area_e[list(marked)] / np.pi)
        else:
            bbox = None; area = 0.0; eq = np.array(())
        levels.append({
            "level": level, "node_count": state.mesh.nn, "element_count": state.mesh.ne,
            "physical_mark_count": len(marked), "physical_marked_area_m2": area,
            "unique_initial_parent_elements_affected_this_level": len(roots),
            "minimum_area_equivalent_size_m": float(np.min(eq)) if eq.size else None,
            "marked_region_bounding_box": bbox,
            "active_tip_hbar_m": active_tip_hbar(state),
            "candidate_visibility": trial_stiffness_visibility(
                state, candidates, da_phys_m=da, crack_band_radius_m=0.0,
            ),
            "counts_by_reason": dict(sorted(reason_counts.items())),
            "counts_by_tip_candidate_reason": [
                {"tip_id": key[0], "candidate_id": key[1], "reason": key[2], "count": value}
                for key, value in sorted(association_counts.items())
            ],
            "monotonic_parent_child_metrics": monotonic,
            "records": records,
        })
        if args.plot and (marked or conformity_elements):
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection
            from matplotlib.patches import Circle
            figure, axes = plt.subplots(figsize=(10, 8))
            marked_set = set(marked)
            focus = marked_set | conformity_elements
            vertices = state.mesh.nodes[state.mesh.elems]
            if focus:
                points = vertices[list(focus)].reshape(-1, 2)
                lo = points.min(axis=0) - 2.0 * contour
                hi = points.max(axis=0) + 2.0 * contour
                axes.set_xlim(lo[0], hi[0]); axes.set_ylim(lo[1], hi[1])
            axes.add_collection(PolyCollection(
                vertices, facecolors="none", edgecolors="#d0d0d0", linewidths=0.2,
            ))
            colors = {
                "candidate_segment_intersection": "#d62728",
                "candidate_crack_normal_span": "#ff7f0e",
                "current_tip_j_support_area": "#1f77b4",
                "candidate_endpoint_j_support_area": "#9467bd",
            }
            reason_by_element = defaultdict(set)
            for record in audit.records:
                reason_by_element[record.element_id].update(record.reasons)
            for reason, color in colors.items():
                ids = sorted(element_id for element_id, reasons in reason_by_element.items() if reason in reasons)
                if ids:
                    axes.add_collection(PolyCollection(
                        vertices[ids], facecolors=color, edgecolors="black",
                        linewidths=0.25, alpha=0.65, label=reason,
                    ))
            if conformity_elements:
                ids = sorted(conformity_elements.difference(marked_set))
                if ids:
                    axes.add_collection(PolyCollection(
                        vertices[ids], facecolors="#7f7f7f", edgecolors="black",
                        linewidths=0.25, alpha=0.45, label="conformity_only",
                    ))
            for tip_id in sorted(state.crack_network.active_tip_ids):
                tip = np.asarray(state.crack_network.branch(tip_id).tip)
                axes.scatter(*tip, marker="*", s=90, color="black", zorder=5)
                axes.add_patch(Circle(tip, contour, fill=False, color="#1f77b4", linewidth=0.8))
                for candidate in sorted(candidates[tip_id], key=lambda item: item.candidate_id):
                    end = tip + da * np.asarray(candidate.direction_xy)
                    axes.plot((tip[0], end[0]), (tip[1], end[1]), color="#d62728", linewidth=1.0)
                    axes.add_patch(Circle(end, contour, fill=False, color="#9467bd", linewidth=0.8))
            axes.set_aspect("equal"); axes.set_title(f"v11 refinement reasons, level {level}")
            axes.set_xlabel("x (m)"); axes.set_ylabel("y (m)"); axes.legend(loc="best", fontsize=7)
            figure.tight_layout()
            figure.savefig(args.out.with_name(f"{args.out.stem}_level{level:02d}.png"), dpi=180)
            plt.close(figure)
        if not marked or level == args.levels:
            break
        refined, lineage = refine_accepted_state(
            state, marked_parent_elements=marked,
            active_tip_ids=state.crack_network.active_tip_ids,
            generation=level + 1, operation_index=level + 1,
            longest_edge_closure=True,
        )
        next_roots = {}
        conformity_elements = {
            int(child)
            for parent, children in lineage.parent_to_child_element_map.items()
            if parent not in set(marked) and len(children) > 1
            for child in children
        }
        for parent, children in lineage.parent_to_child_element_map.items():
            for child in children:
                next_roots[int(child)] = root_lineage[parent]
        root_lineage = next_roots
        parent_records = audit
        parent_map = lineage.parent_to_child_element_map
        state = refined

    payload = {
        "schema": "v11.reason-resolved-physical-marking/1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "da_phys_m": da, "target_resolution_m": target,
        "contour_radius_m": contour,
        "cumulative_mark_operations": cumulative,
        "unique_initial_parent_elements_affected": len(all_roots),
        "unique_level_leaf_mark_pairs": len(all_leaf_ids),
        "levels": levels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = dict(payload)
    summary["levels"] = [
        {key: value for key, value in level.items()
         if key not in {"records", "monotonic_parent_child_metrics"}}
        for level in levels
    ]
    args.out.with_name(args.out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
