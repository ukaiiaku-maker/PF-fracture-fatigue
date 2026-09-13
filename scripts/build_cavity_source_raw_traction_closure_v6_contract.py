#!/usr/bin/env python3
"""Freeze the final V6 D/E closure family before its FEM results are run."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_source_shape_regular_mesh_v5 import (
    ACCEPTANCE_MINIMUM_QUALITY,
    CONSTRUCTION_TARGET_MINIMUM_QUALITY,
    MESH_CONTRACT_ID,
    V6_LOCAL_LEVELS,
    build_shape_regular_source_patch_hole_mesh,
)

OUTPUT = ROOT / "artifacts/v6_cavity_source_raw_traction_closure/v6_contract.json"
V6_CONTRACT_ID = "CAVITY_SOURCE_RAW_TRACTION_CLOSURE_V6"


def build_record():
    levels = []
    for name in ("D", "E"):
        hole = build_shape_regular_source_patch_hole_mesh(
            1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5,
            local_level=name, polygon_sectors=128,
        )
        levels.append({
            "level": name,
            **V6_LOCAL_LEVELS[name],
            "pre_fem_minimum_quality": hole.validation["minimum_quality"],
            "polygon_geometry_fingerprint": hole.validation["polygon_geometry_fingerprint"],
            "discrete_cavity_boundary_fingerprint": hole.validation[
                "discrete_cavity_boundary_fingerprint"
            ],
            "node_count": hole.mesh.nn,
            "element_count": hole.mesh.ne,
        })
    return {
        "schema": "v6.cavity-source-raw-traction-closure-contract/1",
        "contract": V6_CONTRACT_ID,
        "retained_v5_mesh_contract": MESH_CONTRACT_ID,
        "prospectively_frozen_before_D_or_E_fem_evaluation": True,
        "fixed_N_theta": 128,
        "retained_levels": {
            "A": {"first_strip_radial_subdivisions": 2, "raw_traction": 0.14906976345973477},
            "B": {"first_strip_radial_subdivisions": 4, "raw_traction": 0.09144336014017264},
            "C": {"first_strip_radial_subdivisions": 8, "raw_traction": 0.06715193304848284},
        },
        "new_levels": levels,
        "conditional_N256_diagnostic": {
            "raw_traction": 0.038672813574278736,
            "role": "RETAINED_LOW_QUALITY_DIAGNOSTIC_ONLY_NOT_A_QUALIFICATION_LEVEL",
        },
        "construction_target_minimum_quality": CONSTRUCTION_TARGET_MINIMUM_QUALITY,
        "acceptance": {
            "raw_traction_E_max": 0.05,
            "strict_raw_traction_order": "C>D>E",
            "source_tensor_D_to_E_relative_max": 0.05,
            "eta_n_max": 0.03,
            "eta_t_max": 0.025,
            "minimum_quality": ACCEPTANCE_MINIMUM_QUALITY,
            "weak_residual_within_retained_limit": True,
            "bounded_patch_conditioning": True,
            "exact_geometry_and_state_identity": True,
            "physical_equilibrium_observables": True,
        },
        "fixed_across_A_through_E": [
            "N128_polygon", "polygon_fingerprint", "complete_cavity_boundary_fingerprint",
            "source_coordinate", "source_node_incidence", "tangential_boundary_discretization",
            "outer_mesh", "crack_support_geometry", "physical_source_window", "recovery_operator",
        ],
        "maximum_new_levels": 2,
        "additional_level_after_D_or_E": "PROHIBITED",
        "scope": {
            "v1_through_v5_reinterpreted": False,
            "material_or_core_law_changed": False,
            "scientific_tolerance_changed": False,
            "raw_traction_normalization_changed": False,
            "recovery_operator_changed": False,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
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
