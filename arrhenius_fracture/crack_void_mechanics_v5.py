"""Combined V12 sharp-wake and body-fitted explicit-cavity static mechanics."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Any

import numpy as np

from .config import ElasticProperties
from .crack_network_v11 import CrackNetworkState
from .explicit_cavity_v5 import (
    build_explicit_hole_mesh, conform_crack_path, fill_explicit_hole_mesh, solve_static_hole,
)
from .mechanically_separating_sharp_wake_v12 import mechanically_separating_graph_support

SCHEMA = "v12.crack-void-static/5"


def _ray_polygon_intersection(origin, direction, nodes, edges):
    hits = []
    origin = np.asarray(origin, float); direction = np.asarray(direction, float)
    for edge in np.asarray(edges, int):
        a, b = nodes[edge]
        matrix = np.column_stack((direction, -(b - a)))
        det = float(np.linalg.det(matrix))
        if abs(det) <= 1e-15: continue
        ray_t, edge_t = np.linalg.solve(matrix, a - origin)
        if ray_t >= -1e-12 and -1e-12 <= edge_t <= 1.0 + 1e-12:
            hits.append((float(max(ray_t, 0.0)), origin + max(ray_t, 0.0) * direction))
    return min(hits, key=lambda value: value[0]) if hits else None


def _ideal_circle_ray_intersection(origin, direction, center, radius):
    delta = np.asarray(origin, float) - np.asarray(center, float)
    b = 2.0 * float(delta @ direction)
    c = float(delta @ delta) - radius * radius
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0: return None
    roots = [value for value in ((-b - math.sqrt(discriminant)) / 2.0,
                                 (-b + math.sqrt(discriminant)) / 2.0) if value >= -1e-12]
    if not roots: return None
    distance = max(0.0, min(roots))
    return distance, np.asarray(origin, float) + distance * direction


def solve_crack_void_case(*, cavity_center_m=(7.0e-4, 0.0), cavity_radius_m=5.0e-5,
                          boundary_segments=32, radial_layers=12, tip_layer=3,
                          opening_m=4.0e-7, crack_enabled=True, cavity_enabled=True,
                          specimen_width_m=1.0e-3, specimen_height_m=1.0e-3,
                          far_h_m=5.0e-5, material_E_Pa=210e9, material_nu=0.3,
                          residual_stiffness_kappa=1.0e-6, capture_source=False,
                          ligament_ratio: float | None = None,
                          crack_orientation_deg: float = 0.0,
                          crack_root_m: tuple[float, float] | None = None,
                          crack_tip_m: tuple[float, float] | None = None,
                          crack_path_m: tuple[tuple[float, float], ...] | None = None,
                          quality_strategy: str | None = None,
                          geometry_mode: str = "LEGACY_CAVITY_CENTERED_DIAGNOSTIC") -> dict[str, Any]:
    hole = build_explicit_hole_mesh(
        specimen_width_m, specimen_height_m, cavity_center_m, cavity_radius_m,
        far_h_m, boundary_segments, radial_layers_override=radial_layers,
    )
    if not cavity_enabled:
        hole = fill_explicit_hole_mesh(hole)
    support_ids = np.empty(0, dtype=int)
    support_audit = None
    crack_tip = None
    network = None
    conformity_audit = None
    if crack_enabled:
        ntheta = boundary_segments
        if geometry_mode not in ("LEGACY_CAVITY_CENTERED_DIAGNOSTIC", "V3_FIXED_LABORATORY_GEOMETRY"):
            raise ValueError("unknown crack geometry mode")
        if geometry_mode == "V3_FIXED_LABORATORY_GEOMETRY" and crack_path_m is None:
            raise ValueError("V3 qualification requires explicit crack_path_m")
        if crack_path_m is not None and (crack_root_m is not None or crack_tip_m is not None):
            raise ValueError("supply crack_path_m or crack_root_m/crack_tip_m, not both")
        if (crack_root_m is None) != (crack_tip_m is None):
            raise ValueError("crack_root_m and crack_tip_m must be supplied together")
        if crack_path_m is not None:
            hole, conformity_audit = conform_crack_path(hole, crack_path_m)
            path = tuple(tuple(map(float, value)) for value in crack_path_m)
            start_xy, tip_xy = path[0], path[-1]
        elif crack_root_m is not None:
            path = (tuple(map(float, crack_root_m)), tuple(map(float, crack_tip_m)))
            hole, conformity_audit = conform_crack_path(hole, path)
            start_xy, tip_xy = path[0], path[-1]
        else:
            angle = math.radians(180.0 + float(crack_orientation_deg))
            direction = np.array((math.cos(angle), math.sin(angle)))
            if ligament_ratio is None:
                ray_index = ntheta // 2
                outer = radial_layers * ntheta + ray_index
                tip = max(1, min(int(tip_layer), radial_layers - 1)) * ntheta + ray_index
            else:
                center = np.asarray(cavity_center_m, dtype=float)
                desired_tip = center + direction * cavity_radius_m * (1.0 + float(ligament_ratio))
                ray_index = int(round(((180.0 + float(crack_orientation_deg)) % 360.0) / 360.0 * ntheta)) % ntheta
                ray_nodes = np.arange(radial_layers + 1) * ntheta + ray_index
                tip = int(ray_nodes[np.argmin(np.linalg.norm(hole.mesh.nodes[ray_nodes] - desired_tip, axis=1))])
                outer = int(radial_layers * ntheta + ray_index)
            start_xy = tuple(map(float, hole.mesh.nodes[outer]))
            tip_xy = tuple(map(float, hole.mesh.nodes[tip]))
            path = (start_xy, tip_xy)
        if quality_strategy is not None:
            from .quality_constrained_mesh_v1 import constrained_quality_mesh, fixed_path_constraints
            fixed, protected = fixed_path_constraints(hole.mesh, path)
            mesh, quality_audit = constrained_quality_mesh(hole.mesh, fixed_nodes=fixed,
                protected_edges=protected, strategy=quality_strategy)
            hole = replace(hole, mesh=mesh)
            conformity_audit = dict(conformity_audit or {}, quality_constrained_mesh_v1=quality_audit)
        network = CrackNetworkState.one_tip(path)
        support_ids, support_audit = mechanically_separating_graph_support(hole.mesh, network)
        crack_tip = tip_xy
    mask = np.zeros(hole.mesh.ne, dtype=bool)
    mask[support_ids] = True
    result = solve_static_hole(
        hole, opening_m, ElasticProperties(E=material_E_Pa, nu=material_nu),
        crack_tip_m=crack_tip,
        element_kill_mask=mask if crack_enabled else None,
        residual_stiffness_kappa=residual_stiffness_kappa, capture_source=capture_source,
    )
    if len(hole.cavity_edges):
        edge_vectors = hole.mesh.nodes[hole.cavity_edges[:, 1]] - hole.mesh.nodes[hole.cavity_edges[:, 0]]
        perimeter = float(np.sum(np.linalg.norm(edge_vectors, axis=1)))
        xy = hole.mesh.nodes[hole.prescribed_polygon_nodes]
        area = float(abs(0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))))
    else:
        perimeter = area = 0.0
    probes = {
        "sigma_xx_min_Pa": float(np.min(result.sigma_gp[0])),
        "sigma_xx_max_Pa": float(np.max(result.sigma_gp[0])),
        "sigma_yy_min_Pa": float(np.min(result.sigma_gp[1])),
        "sigma_yy_max_Pa": float(np.max(result.sigma_gp[1])),
        "sigma_xy_abs_max_Pa": float(np.max(np.abs(result.sigma_gp[2]))),
    }
    center = np.asarray(cavity_center_m, dtype=float)
    realized_root = None if network is None else np.asarray(network.branches[0].root, dtype=float)
    realized_tip = None if network is None else np.asarray(network.branches[0].tip, dtype=float)
    if realized_tip is None or not len(hole.cavity_edges):
        ligament_m = polygon_intersection = ideal_intersection = orientation = None
        ray_intersects = None; shortest = axial = offset = None
        tip_h = cavity_h_max = cavity_h_median = ligament_h_max = ligament_h_median = None
    else:
        path = np.asarray(network.branches[0].path, float)
        segment = path[-1] - path[-2]
        tangent = segment / np.linalg.norm(segment); normal = np.array((-tangent[1], tangent[0]))
        orientation = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
        delta = center - realized_tip
        axial = float(delta @ tangent); offset = float(delta @ normal)
        polygon_hit = _ray_polygon_intersection(realized_tip, tangent, hole.mesh.nodes, hole.cavity_edges)
        ideal_hit = _ideal_circle_ray_intersection(realized_tip, tangent, center, float(cavity_radius_m))
        ray_intersects = polygon_hit is not None
        ligament_m = None if polygon_hit is None else polygon_hit[0]
        polygon_intersection = None if polygon_hit is None else polygon_hit[1].tolist()
        ideal_intersection = None if ideal_hit is None else ideal_hit[1].tolist()
        cavity_segments = hole.mesh.nodes[hole.cavity_edges]
        shortest = min(float(np.linalg.norm(realized_tip - (a + np.clip((realized_tip-a)@(b-a)/max((b-a)@(b-a),1e-300),0,1)*(b-a)))) for a,b in cavity_segments)
        cavity_lengths = np.linalg.norm(cavity_segments[:, 1] - cavity_segments[:, 0], axis=1)
        cavity_h_max, cavity_h_median = float(np.max(cavity_lengths)), float(np.median(cavity_lengths))
        incident = np.asarray([e for e in hole.mesh.elems if np.any(e == int(np.argmin(np.linalg.norm(hole.mesh.nodes-realized_tip, axis=1))))])
        tip_edges = np.linalg.norm(hole.mesh.nodes[incident[:, [1,2,0]]] - hole.mesh.nodes[incident[:, [0,1,2]]], axis=2)
        tip_h = float(np.median(tip_edges))
        if polygon_hit is None:
            ligament_h_max = ligament_h_median = None
        else:
            centroids = hole.mesh.nodes[hole.mesh.elems].mean(axis=1)
            along = (centroids - realized_tip) @ tangent
            lateral = np.abs((centroids - realized_tip) @ normal)
            chosen = hole.mesh.elems[(along >= 0) & (along <= ligament_m) & (lateral <= cavity_h_max)]
            lengths = np.linalg.norm(hole.mesh.nodes[chosen[:, [1,2,0]]] - hole.mesh.nodes[chosen[:, [0,1,2]]], axis=2).ravel()
            ligament_h_max = float(np.max(lengths)); ligament_h_median = float(np.median(lengths))
    # Cavity resolution exists independently of the presence of a crack.
    if len(hole.cavity_edges):
        cavity_lengths = np.linalg.norm(hole.mesh.nodes[hole.cavity_edges[:, 1]] -
                                       hole.mesh.nodes[hole.cavity_edges[:, 0]], axis=1)
        cavity_h_max, cavity_h_median = float(np.max(cavity_lengths)), float(np.median(cavity_lengths))
    return {
        **({"source_capture": result.source_capture} if capture_source else {}),
        "schema": SCHEMA,
        "configuration": {
            "cavity_center_m": list(cavity_center_m), "cavity_radius_m": cavity_radius_m,
            "boundary_segments": boundary_segments, "radial_layers": radial_layers,
            "tip_layer": tip_layer, "opening_m": opening_m,
            "ligament_ratio_requested": ligament_ratio,
            "crack_root_m_requested": None if crack_root_m is None else list(crack_root_m),
            "crack_tip_m_requested": None if crack_tip_m is None else list(crack_tip_m),
            "crack_path_m_requested": None if crack_path_m is None else [list(v) for v in crack_path_m],
            "geometry_mode": geometry_mode,
            "crack_orientation_deg": crack_orientation_deg,
            "crack_enabled": crack_enabled, "cavity_enabled": cavity_enabled,
            "support_selection": "exact_v12_only", "centroid_band_fallback": False,
        },
        "observables": {
            "reaction_top_N_per_m": result.reaction_top_N_per_m,
            "reaction_bottom_N_per_m": result.reaction_bottom_N_per_m,
            "compliance_m2_per_N": result.compliance_m2_per_N,
            "stored_energy_J_per_m": result.stored_energy_J_per_m,
            "free_residual_norm_N_per_m": result.free_residual_norm_N_per_m,
            "cavity_traction_l2_normalized": result.traction_l2_normalized,
            "cavity_traction_normal_l2_normalized": result.traction_normal_l2_normalized,
            "cavity_traction_tangential_l2_normalized": result.traction_tangential_l2_normalized,
            "cavity_traction_resultant_normalized": list(result.traction_resultant_normalized),
            "cavity_traction_moment_normalized": result.traction_moment_normalized,
            "weak_cavity_boundary_residual_relative": result.weak_cavity_residual_relative,
            "assembled_cavity_node_residual_relative": result.weak_cavity_residual_relative,
            "cavity_traction_l2_dimensional_Pa_sqrt_m": result.traction_l2_dimensional_Pa_sqrt_m,
            "cavity_traction_normal_l2_dimensional_Pa_sqrt_m": result.traction_normal_l2_dimensional_Pa_sqrt_m,
            "cavity_traction_tangential_l2_dimensional_Pa_sqrt_m": result.traction_tangential_l2_dimensional_Pa_sqrt_m,
            "nominal_remote_stress_Pa": result.nominal_remote_stress_Pa,
            "traction_diagnostic_cavity_perimeter_m": result.cavity_perimeter_m,
            "cavity_edge_traction_records": [dict(row) for row in result.cavity_edge_traction_records],
            "hoop_stress_concentration": result.hoop_stress_concentration,
            "crack_tip_sigma_yy_Pa": result.crack_tip_sigma_yy_Pa,
            "symmetry_error": result.symmetry_error,
            "cavity_area_m2": area,
            "cavity_perimeter_m": perimeter,
            "mesh_nodes": int(hole.mesh.nn), "mesh_elements": int(hole.mesh.ne),
            "mesh_minimum_quality": float(hole.validation.get("minimum_quality", math.nan)),
            "mesh_maximum_aspect_ratio": float(hole.validation.get("maximum_aspect_ratio", math.nan)),
            "internal_boundary_components": int(hole.validation.get("actual_internal_components", 0)),
            "v12_support_elements": int(len(support_ids)),
            "v12_support_certified": bool(support_audit.certified) if support_audit else None,
            "crack_graph_length_m": 0.0 if network is None else float(network.total_physical_crack_length_m),
            "crack_root_m": None if realized_root is None else realized_root.tolist(),
            "crack_tip_m": None if realized_tip is None else realized_tip.tolist(),
            "cavity_center_m": list(map(float, cavity_center_m)),
            "ideal_circle_intersection_m": ideal_intersection,
            "realized_polygon_intersection_m": polygon_intersection,
            "polygonization_intersection_error_m": None if ideal_intersection is None or polygon_intersection is None else float(np.linalg.norm(np.asarray(ideal_intersection)-np.asarray(polygon_intersection))),
            "ray_intersects_polygon": ray_intersects,
            "ligament_length_m": ligament_m,
            "ligament_over_radius": None if ligament_m is None else ligament_m / float(cavity_radius_m),
            "shortest_tip_to_cavity_m": shortest,
            "axial_separation_over_R": None if axial is None else axial / float(cavity_radius_m),
            "signed_normal_offset_over_R": None if offset is None else offset / float(cavity_radius_m),
            "R_over_h_cavity_max": None if cavity_h_max is None else float(cavity_radius_m) / cavity_h_max,
            "R_over_h_cavity_median": None if cavity_h_median is None else float(cavity_radius_m) / cavity_h_median,
            "ligament_over_h_max": None if ligament_m is None else ligament_m / ligament_h_max,
            "ligament_over_h_median": None if ligament_m is None else ligament_m / ligament_h_median,
            "tip_h_m": tip_h,
            "cavity_boundary_edge_lengths_m": [] if realized_tip is None or not len(hole.cavity_edges) else cavity_lengths.tolist(),
            "crack_orientation_realized_deg": orientation,
            **probes,
        },
        "support_audit": None if support_audit is None else asdict(support_audit),
        "geometry_conformity_audit": conformity_audit,
    }
