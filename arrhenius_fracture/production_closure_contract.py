"""Prospective contract for the bounded one-void production closure.

This module is deliberately result-free.  It freezes the final D/E geometry,
the scientific limits, and the independent traction-verification fallback
before a new D/E or central DBTT evaluation is performed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType

from .cavity_source_recovery_v3 import MAXIMUM_DESIGN_CONDITION
from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES


CONTRACT_ID = "UNIFIED_2D_ONE_VOID_PRODUCTION_CLOSURE"
SCHEMA = "unified-2d-one-void.production-closure-contract/1"
FALLBACK_OPERATOR_ID = "EQUILIBRATED_BOUNDARY_TRACTION_RECOVERY_V1"


@dataclass(frozen=True)
class LocalMeshLevel:
    name: str
    first_strip_radial_subdivisions: int


FINAL_LOCAL_LEVELS = (
    LocalMeshLevel("D", 16),
    LocalMeshLevel("E", 32),
)

ACCEPTANCE_LIMITS = MappingProxyType({
    "source_tensor_relative_change": 0.05,
    "eta_n": 0.03,
    "eta_t": 0.025,
    "minimum_mesh_quality": 0.05,
    "construction_minimum_mesh_quality": 0.10,
    "raw_adjacent_cst_traction": 0.05,
    "equilibrated_boundary_traction": 0.05,
    "equilibrated_traction_mesh_convergence": 0.05,
    "weak_residual": SCIENTIFIC_ACCEPTANCE_TOLERANCES["free_residual_relative"],
    "patch_condition": MAXIMUM_DESIGN_CONDITION,
    "reaction_balance": SCIENTIFIC_ACCEPTANCE_TOLERANCES["reaction_balance_relative"],
})


def contract_record() -> dict:
    """Return the frozen, machine-readable contract without numerical results."""
    return {
        "schema": SCHEMA,
        "contract": CONTRACT_ID,
        "prospectively_frozen_before_new_D_E_or_DBTT_evaluation": True,
        "physical_equilibrium_observation_required": True,
        "final_local_levels": [asdict(level) for level in FINAL_LOCAL_LEVELS],
        "acceptance_limits": dict(ACCEPTANCE_LIMITS),
        "fallback": {
            "operator": FALLBACK_OPERATOR_ID,
            "role": "INDEPENDENT_TRACTION_VERIFICATION_ONLY",
            "accepted_element_stresses": True,
            "constrained_weighted_least_squares": True,
            "zero_body_force_local_equilibrium": True,
            "internal_traction_continuity": "SINGLE_CONTINUOUS_AIRY_STRESS_FIELD",
            "zero_boundary_traction_imposed": False,
            "traction_evaluated_on_actual_cavity_boundary": True,
            "rotation_covariance_required": True,
            "edge_order_invariance_required": True,
        },
        "fixed_geometry": {
            "polygon_sectors": 128,
            "complete_cavity_boundary": "V5_N128_SOURCE_CONFORMING_POLYGON",
            "physical_source_window": "V3_FIXED_PHYSICAL_ARC_WINDOW",
            "additional_level_after_E": "PROHIBITED",
        },
        "immutable_model_boundary": {
            "fracture_or_void_kinetics_changed": False,
            "material_bundle_changed": False,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
            "scientific_tolerance_changed": False,
        },
    }


__all__ = [
    "ACCEPTANCE_LIMITS", "CONTRACT_ID", "FALLBACK_OPERATOR_ID",
    "FINAL_LOCAL_LEVELS", "LocalMeshLevel", "SCHEMA", "contract_record",
]
