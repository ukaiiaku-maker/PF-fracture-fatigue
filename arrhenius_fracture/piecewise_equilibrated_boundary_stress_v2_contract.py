"""Prospective contract for the final local traction verifier.

This module is deliberately result-free.  It freezes the operator, physical
patch, and scientific limits before the central D/E states are evaluated.
"""
from __future__ import annotations

from types import MappingProxyType

from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES


CONTRACT_ID = "PIECEWISE_EQUILIBRATED_BOUNDARY_STRESS_V2_QUALIFICATION"
OPERATOR_ID = "PIECEWISE_EQUILIBRATED_BOUNDARY_STRESS_V2"
SCHEMA = "unified-2d-one-void.piecewise-equilibrated-stress-contract/1"

ACCEPTANCE_LIMITS = MappingProxyType({
    "normalized_equilibrated_boundary_traction": 0.05,
    "traction_mesh_relative_change": 0.05,
    "minimum_mesh_quality": 0.05,
    "maximum_scaled_condition": 1.0e8,
    "maximum_stress_fit_residual": 0.35,
    "maximum_equilibrium_residual": 1.0e-12,
    "maximum_interface_traction_jump_relative": 1.0e-8,
    "maximum_resultant_residual_relative": 1.0e-8,
    "maximum_moment_residual_relative": 1.0e-8,
    "weak_fem_residual": SCIENTIFIC_ACCEPTANCE_TOLERANCES[
        "free_residual_relative"
    ],
})


def contract_record() -> dict:
    return {
        "schema": SCHEMA,
        "contract": CONTRACT_ID,
        "operator": OPERATOR_ID,
        "prospectively_frozen_before_central_D_E_evaluation": True,
        "stress_space": "ELEMENT_LOCAL_AFFINE_SYMMETRIC_DIVERGENCE_FREE",
        "fit_target": "ACCEPTED_CST_STRESS",
        "physical_patch": "EXACT_V3_FIXED_PHYSICAL_SOURCE_PATCH",
        "constraints": {
            "local_in_plane_equilibrium": "EXACT_BY_PARAMETERIZATION",
            "tensor_symmetry": "EXACT_BY_PARAMETERIZATION",
            "internal_normal_traction_continuity": "EDGE_ENDPOINT_EQUALITY",
            "patch_resultant_balance": True,
            "patch_moment_balance": True,
            "zero_cavity_traction_imposed": False,
        },
        "boundary_evaluation": "ACTUAL_OWNED_CAVITY_EDGES_AFTER_SOLVE",
        "qualification_cases": [
            "constant_stress", "affine_self_equilibrated_stress",
            "rigid_rotation", "reversed_normals_and_edge_order", "Kirsch",
            "reflected_cavity_geometry", "two_shape_regular_refinement_levels",
        ],
        "acceptance_limits": dict(ACCEPTANCE_LIMITS),
        "central_levels": ["D", "E"],
        "additional_production_mesh_level": "PROHIBITED",
        "raw_CST_traction_role": "RETAINED_NON_AUTHORITATIVE_DIAGNOSTIC",
        "immutable_model_boundary": {
            "shared_v13_fracture_core_changed": False,
            "material_or_constitutive_model_changed": False,
            "fracture_or_void_kinetics_changed": False,
            "source_tensor_definition_changed": False,
            "r_tip_or_R_void_changed": False,
            "threshold_or_RNG_changed": False,
            "traction_limit_changed": False,
            "qualified_D_E_geometry_changed": False,
            "physical_equilibrium_observation_changed": False,
        },
    }


__all__ = ["ACCEPTANCE_LIMITS", "CONTRACT_ID", "OPERATOR_ID", "SCHEMA", "contract_record"]
