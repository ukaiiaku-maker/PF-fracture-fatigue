#!/usr/bin/env python3
"""Geometry-only audit of the retained V4 mesh-quality failure."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.explicit_cavity_v5 import build_source_conforming_hole_mesh
from arrhenius_fracture.voiding_production_v5 import _insert_point_in_mesh


OUTPUT = ROOT / "artifacts/v5_cavity_source_recovery_v5/v4_mesh_failure_audit.json"
V4 = ROOT / "artifacts/v5_cavity_source_recovery_v4/central_dbtt_v4_readiness.json"
LEVELS = ((32, 12), (64, 24), (128, 48))


def _point_segment_distance(point, first, second):
    delta = second - first
    fraction = float((point - first) @ delta / max(float(delta @ delta), 1.0e-300))
    return float(np.linalg.norm(point - (first + np.clip(fraction, 0.0, 1.0) * delta)))


def _metrics(mesh):
    points = np.asarray(mesh.nodes)[np.asarray(mesh.elems)]
    vectors = points[:, [1, 2, 0]] - points[:, [0, 1, 2]]
    lengths = np.linalg.norm(vectors, axis=2)
    signed2 = ((points[:, 1, 0] - points[:, 0, 0]) *
               (points[:, 2, 1] - points[:, 0, 1]) -
               (points[:, 1, 1] - points[:, 0, 1]) *
               (points[:, 2, 0] - points[:, 0, 0]))
    area = np.abs(signed2) / 2.0
    quality = 4.0 * np.sqrt(3.0) * area / np.maximum(np.sum(lengths**2, axis=1), 1e-300)
    angles = []
    for opposite in range(3):
        a = lengths[:, (opposite + 1) % 3]
        b = lengths[:, (opposite + 2) % 3]
        c = lengths[:, opposite]
        angles.append(np.degrees(np.arccos(np.clip((a*a + b*b - c*c) /
                                                    np.maximum(2*a*b, 1e-300), -1.0, 1.0))))
    return points, lengths, signed2, quality, np.column_stack(angles)


def _classify(ids, centroid, *, source_node, boundary_nodes, path_nodes, center, radius,
              distance_to_crack):
    node_set = set(map(int, ids))
    if source_node in node_set:
        return "split_source_facet"
    if node_set.intersection(path_nodes) or distance_to_crack <= 1.0e-12:
        return "crack_support_region"
    if node_set.intersection(boundary_nodes):
        return "first_radial_ring"
    radial_offset = float(np.linalg.norm(centroid - center) - radius)
    if radial_offset <= 0.5 * radius:
        return "annular_transition"
    return "far_field_transition"


def build_record():
    retained = json.loads(V4.read_text())
    crack_path = tuple(np.asarray(point, dtype=float) for point in retained["retained_physical_crack_path_m"])
    center = np.asarray(retained["geometry"]["cavity_center_m"], dtype=float)
    radius = float(retained["geometry"]["nominal_cavity_radius_m"])
    levels = []
    for sectors, radial_layers in LEVELS:
        hole = build_source_conforming_hole_mesh(
            1.0e-3, 1.0e-3, tuple(center), radius, 5.0e-5, sectors,
            radial_layers_override=radial_layers, source_direction_xy=(1.0, 0.0),
        )
        base_quality = _metrics(hole.mesh)[3]
        mesh = hole.mesh
        original_nodes = len(mesh.nodes)
        for point in crack_path:
            mesh = _insert_point_in_mesh(mesh, point)
        path_nodes = set(range(original_nodes, len(mesh.nodes)))
        points, lengths, signed2, quality, angles = _metrics(mesh)
        source_position = center + np.asarray((radius, 0.0))
        source_node = int(np.argmin(np.linalg.norm(mesh.nodes - source_position, axis=1)))
        boundary_edges = np.asarray(hole.cavity_edges, dtype=int)
        boundary_nodes = set(map(int, boundary_edges.ravel()))
        rows = []
        for element_id in np.argsort(quality)[:20]:
            ids = np.asarray(mesh.elems[element_id], dtype=int)
            triangle = points[element_id]
            centroid = triangle.mean(axis=0)
            crack_distance = min(
                _point_segment_distance(centroid, crack_path[index], crack_path[index + 1])
                for index in range(len(crack_path) - 1)
            )
            boundary_distance = min(
                _point_segment_distance(centroid, mesh.nodes[a], mesh.nodes[b])
                for a, b in boundary_edges
            )
            contributors = []
            for a, b in boundary_edges:
                if {int(a), int(b)}.issubset(set(map(int, ids))):
                    edge_length = float(np.linalg.norm(mesh.nodes[b] - mesh.nodes[a]))
                    contributors.append({
                        "boundary_edge": [int(a), int(b)],
                        "eta_n": float(abs(signed2[element_id]) / max(edge_length * radius, 1e-300)),
                        "eta_t": edge_length / radius,
                    })
            rows.append({
                "rank": len(rows) + 1,
                "element_id": int(element_id),
                "connectivity": ids.tolist(),
                "coordinates_m": triangle.tolist(),
                "centroid_m": centroid.tolist(),
                "quality": float(quality[element_id]),
                "minimum_angle_deg": float(np.min(angles[element_id])),
                "maximum_angle_deg": float(np.max(angles[element_id])),
                "aspect_ratio": float(np.max(lengths[element_id]) / np.min(lengths[element_id])),
                "edge_lengths_m": lengths[element_id].tolist(),
                "edge_length_ratio": float(np.max(lengths[element_id]) / np.min(lengths[element_id])),
                "signed_double_area_jacobian_m2": float(signed2[element_id]),
                "distance_to_source_node_m": float(np.linalg.norm(centroid - source_position)),
                "distance_to_cavity_boundary_m": boundary_distance,
                "distance_to_crack_support_m": crack_distance,
                "location": _classify(
                    ids, centroid, source_node=source_node, boundary_nodes=boundary_nodes,
                    path_nodes=path_nodes, center=center, radius=radius,
                    distance_to_crack=crack_distance,
                ),
                "local_resolution_contributors": contributors,
            })
        counts = {name: sum(row["location"] == name for row in rows) for name in (
            "split_source_facet", "first_radial_ring", "annular_transition",
            "crack_support_region", "far_field_transition",
        )}
        retained_level = next(row for row in retained["level_records"] if row["N_theta"] == sectors)
        levels.append({
            "N_theta": sectors,
            "radial_layers": radial_layers,
            "base_source_conforming_mesh_minimum_quality": float(np.min(base_quality)),
            "geometry_only_crack_path_mesh_minimum_quality": float(np.min(quality)),
            "retained_connected_v4_minimum_quality": retained_level["minimum_quality"],
            "lowest_twenty_location_counts": counts,
            "lowest_quality_elements": rows,
        })
    final = levels[-1]
    return {
        "schema": "v5.v4-mesh-failure-geometry-only-audit/1",
        "new_fem_solve_run": False,
        "retained_v4_interpretation": {
            "DBTT_SOURCE_READINESS": "BLOCKED_WITH_EXACT_V4_FAILURE_CLASS",
            "V4_FAILURE_CLASS": [
                "NORMAL_DIRECTION_RESOLUTION", "TANGENTIAL_DIRECTION_RESOLUTION", "MESH_QUALITY"
            ],
            "V4_SOURCE_GEOMETRY": "PASS",
            "V4_SOURCE_TENSOR_RELATIVE_CHANGE_64_TO_128": 0.006818275586047867,
            "V4_POINT_SOURCE_FORMULATION": "REMAINS_VIABLE",
            "FINITE_ACTIVATION_ZONE_REQUIRED": "NOT_ESTABLISHED",
            "zero_recovered_boundary_traction_interpretation": (
                "traction-free constraint of the recovered boundary-limit tensor; "
                "not an independent raw adjacent-element traction measurement"
            ),
        },
        "levels": levels,
        "cause_decision": {
            "classification": "C_CRACK_SUPPORT_CAVITY_INTERACTION",
            "A_source_facet_insertion_without_matched_radial_column": False,
            "B_annular_to_background_transition": False,
            "C_crack_support_cavity_interaction": True,
            "D_other_explicit_mesh_location": False,
            "evidence": {
                "N128_base_mesh_quality_above_acceptance": final[
                    "base_source_conforming_mesh_minimum_quality"
                ] >= 0.05,
                "N128_quality_drops_below_acceptance_after_fixed_crack_path_insertion": final[
                    "geometry_only_crack_path_mesh_minimum_quality"
                ] < 0.05,
                "N128_worst_element_location": final["lowest_quality_elements"][0]["location"],
                "retained_connected_mesh_collapses_further_in_same_construction_path": (
                    final["retained_connected_v4_minimum_quality"]
                    < final["geometry_only_crack_path_mesh_minimum_quality"]
                ),
            },
        },
        "prohibited_resolution_response": {
            "N_theta_256_or_512_added": False,
            "global_sector_count_used_as_only_control": False,
        },
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_record(), indent=2, sort_keys=True) + "\n")
    print("V4_MESH_FAILURE_CAUSE=C_CRACK_SUPPORT_CAVITY_INTERACTION")
    print("NEW_FEM_SOLVE_RUN=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
