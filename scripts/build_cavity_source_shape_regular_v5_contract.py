#!/usr/bin/env python3
"""Freeze the V5 shape-regular mesh contract before central evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_source_shape_regular_mesh_v5 import (
    ACCEPTANCE_MINIMUM_QUALITY, CONSTRUCTION_TARGET_MINIMUM_QUALITY, LOCAL_LEVELS,
    MESH_CONTRACT_ID, build_shape_regular_source_patch_hole_mesh,
)

OUTPUT = ROOT / "artifacts/v5_cavity_source_recovery_v5/v5_mesh_contract.json"


def build_record():
    levels = []
    for name in ("A", "B", "C"):
        hole = build_shape_regular_source_patch_hole_mesh(
            1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5, local_level=name,
        )
        levels.append({
            "level": name,
            **LOCAL_LEVELS[name],
            "pre_fem_minimum_quality": hole.validation["minimum_quality"],
            "polygon_geometry_fingerprint": hole.validation["polygon_geometry_fingerprint"],
            "discrete_cavity_boundary_fingerprint": hole.validation[
                "discrete_cavity_boundary_fingerprint"
            ],
            "node_count": hole.mesh.nn,
            "element_count": hole.mesh.ne,
        })
    return {
        "schema": "v5.cavity-source-shape-regular-local-patch-contract/1",
        "mesh_contract": MESH_CONTRACT_ID,
        "prospectively_frozen_before_central_dbtt_evaluation": True,
        "maximum_mesh_construction_strategies": 2,
        "selected_construction_strategy": "DETERMINISTIC_TWO_TO_ONE_STRUCTURED_LOCAL_PATCH",
        "construction_target_minimum_quality": CONSTRUCTION_TARGET_MINIMUM_QUALITY,
        "acceptance_minimum_quality": ACCEPTANCE_MINIMUM_QUALITY,
        "angular_geometry_family": {
            "required_levels": [64, 128],
            "conditional_level_256": "ONLY_IF_64_TO_128_FAILS",
            "selected_fixed_polygon_sectors_for_local_family": 128,
            "conditional_level_256_role": (
                "BOUNDED_ANGULAR_DIAGNOSTIC_ONLY; DOES_NOT_CHANGE_THE_FROZEN_LOCAL_FAMILY"
            ),
            "matched_local_resolution": True,
            "matched_boundary_node_count": 512,
        },
        "fixed_acceptance": {
            "tensor_relative_max": 0.05,
            "raw_normalized_traction_max": 0.05,
            "eta_n_max": 0.03,
            "eta_t_max": 0.025,
            "minimum_quality": 0.05,
            "bounded_patch_conditioning": True,
            "fixed_physical_window_identity": True,
            "fixed_discrete_cavity_boundary_fingerprint": True,
            "accepted_input_state_unchanged_on_noncertification": True,
        },
        "local_levels": levels,
        "diagnostic_separation": [
            "constrained_boundary_limit_tensor",
            "raw_adjacent_element_traction",
            "assembled_weak_cavity_boundary_residual",
            "constrained_fit_residual",
            "patch_rank_and_condition",
            "local_and_global_source_tensor",
            "eta_n_and_eta_t",
            "local_and_global_minimum_quality",
        ],
        "scope": {
            "material_or_core_law_changed": False,
            "scientific_tolerance_changed": False,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
            "finite_activation_zone_observable_derived": False,
            "oracle_generated": False,
            "paired_trajectories_run": False,
            "fatigue_started": False,
        },
    }


def main():
    OUTPUT.write_text(json.dumps(build_record(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
