"""Combined V12 sharp-wake and body-fitted explicit-cavity static mechanics."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import numpy as np

from .config import ElasticProperties
from .crack_network_v11 import CrackNetworkState
from .explicit_cavity_v5 import build_explicit_hole_mesh, fill_explicit_hole_mesh, solve_static_hole
from .mechanically_separating_sharp_wake_v12 import mechanically_separating_graph_support

SCHEMA = "v12.crack-void-static/5"


def solve_crack_void_case(*, cavity_center_m=(7.0e-4, 0.0), cavity_radius_m=5.0e-5,
                          boundary_segments=32, radial_layers=12, tip_layer=3,
                          opening_m=4.0e-7, crack_enabled=True, cavity_enabled=True,
                          ligament_ratio: float | None = None,
                          crack_orientation_deg: float = 0.0) -> dict[str, Any]:
    hole = build_explicit_hole_mesh(
        1.0e-3, 1.0e-3, cavity_center_m, cavity_radius_m,
        5.0e-5, boundary_segments, radial_layers_override=radial_layers,
    )
    if not cavity_enabled:
        hole = fill_explicit_hole_mesh(hole)
    support_ids = np.empty(0, dtype=int)
    support_audit = None
    crack_tip = None
    network = None
    if crack_enabled:
        ntheta = boundary_segments
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
        network = CrackNetworkState.one_tip((start_xy, tip_xy))
        support_ids, support_audit = mechanically_separating_graph_support(hole.mesh, network)
        crack_tip = tip_xy
    mask = np.zeros(hole.mesh.ne, dtype=bool)
    mask[support_ids] = True
    result = solve_static_hole(
        hole, opening_m, ElasticProperties(E=210e9, nu=0.3),
        crack_tip_m=crack_tip,
        element_kill_mask=mask if crack_enabled else None,
        residual_stiffness_kappa=1.0e-6,
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
    return {
        "schema": SCHEMA,
        "configuration": {
            "cavity_center_m": list(cavity_center_m), "cavity_radius_m": cavity_radius_m,
            "boundary_segments": boundary_segments, "radial_layers": radial_layers,
            "tip_layer": tip_layer, "opening_m": opening_m,
            "ligament_ratio_requested": ligament_ratio,
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
            "hoop_stress_concentration": result.hoop_stress_concentration,
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
            **probes,
        },
        "support_audit": None if support_audit is None else asdict(support_audit),
    }
