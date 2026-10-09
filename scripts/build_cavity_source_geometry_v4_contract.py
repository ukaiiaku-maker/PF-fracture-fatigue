#!/usr/bin/env python3
"""Write the prospective V4 geometry contract without evaluating DBTT."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_source_conforming_geometry_v4 import (
    ETA_N_MAX,
    ETA_T_MAX,
    GEOMETRY_ID,
    MINIMUM_MESH_QUALITY,
    TENSOR_RELATIVE_TOLERANCE,
    TRACTION_RESIDUAL_TOLERANCE,
    radius_convention_from_hole,
)
from arrhenius_fracture.explicit_cavity_v5 import build_source_conforming_hole_mesh


OUTPUT = ROOT / "artifacts/v5_cavity_source_recovery_v4/v4_geometry_contract.json"


def build_record():
    hole = build_source_conforming_hole_mesh(
        1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5, 32,
        radial_layers_override=12, source_direction_xy=(1.0, 0.0),
    )
    radius = radius_convention_from_hole(hole).as_record()
    return {
        "schema": "v5.cavity-source-conforming-geometry-contract/4",
        "geometry_contract": GEOMETRY_ID,
        "prospectively_frozen_before_central_dbtt_evaluation": True,
        "retained_v1_v2_v3_files_modified": False,
        "radius_convention": radius,
        "source_boundary_construction": {
            "polygon_type": "regular_circumscribed",
            "source_ray_xy": [1.0, 0.0],
            "source_facet_centered_on_ray": True,
            "source_node_location": "facet_midpoint_at_nominal_circle_radius",
            "near_and_far_nodes_inserted": True,
            "facet_geometry_changed_by_split": False,
            "required_source_degree": 2,
            "required_source_collinearity": True,
            "required_boundary_owner_count_per_incident_edge": 1,
        },
        "recovery": {
            "operator": "CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
            "fixed_window_reused_unchanged": True,
            "v3_coordinate_check_relaxed": False,
            "unexpected_programming_exceptions_propagate": True,
            "scientific_noncertification_fails_closed": True,
        },
        "acceptance_gates": {
            "tensor_relative_max": TENSOR_RELATIVE_TOLERANCE,
            "normalized_cavity_traction_max": TRACTION_RESIDUAL_TOLERANCE,
            "eta_n_max": ETA_N_MAX,
            "eta_t_max": ETA_T_MAX,
            "minimum_mesh_quality": MINIMUM_MESH_QUALITY,
            "bounded_patch_conditioning": True,
            "maximum_angular_levels": 3,
            "angular_levels": [32, 64, 128],
        },
        "length_ledger_conventions": {
            "fractured_ligament_length_m": "crack_tip_to_actual_near_source_node",
            "active_front_coordinate_advance_m": "same_physical_ligament_length",
            "physical_active_front_travel_m": "same_physical_ligament_length",
            "projected_fractured_length_m": "laboratory_x_projection_to_actual_near_source_node",
            "projected_front_advance_m": "same_laboratory_x_projection",
            "preexisting_void_free_span_m": "actual_near_to_far_source_node_chord",
            "connected_void_free_span_m": "same_actual_source_node_chord",
            "projected_connected_void_free_span_m": "laboratory_x_projection_of_source_node_chord",
            "connected_free_surface_extent_m": "actual_polygon_boundary_arc_between_source_nodes",
        },
        "scope": {
            "cavity_area_and_inventory_use_nominal_circle": True,
            "void_kinetics_use_nominal_circle_radius": True,
            "polygon_used_only_as_fem_boundary": True,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
            "scientific_tolerances_changed": False,
            "finite_activation_zone_observable_derived": False,
            "oracle_generated": False,
            "paired_trajectories_run": False,
            "fatigue_started": False,
        },
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_record(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
