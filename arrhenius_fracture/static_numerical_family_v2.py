"""Prospective V2 static-family contract and retained-V1 failure atlas.

The atlas is diagnostic postprocessing.  It never changes a retained row or
turns a V1 failure into a V2 execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS

SCHEMA = "v5.one-void-static-family-v2/1"
FAILURE_ORDER = (
    "two_quality_valid_fine_levels",
    "closed_cavity_and_no_overlap",
    "independent_intact_cut",
    "unique_live_boundary_edge_owners",
    "free_residual_relative",
    "reaction_balance",
    "energy_identity",
    "reaction_stable_limit",
    "compliance_stable_limit",
    "energy_stable_limit",
    "reaction_fine_accuracy",
    "compliance_fine_accuracy",
    "energy_fine_accuracy",
    "fine_raw_full_boundary_traction",
    "fixed_tip_stable_limit",
    "fixed_tip_fine_accuracy",
    "recovered_fixed_arc_stable_limit",
    "recovered_fixed_arc_fine_accuracy",
)
FAILURE_CLASS = {
    "two_quality_valid_fine_levels": "INSUFFICIENT_QUALITY_VALID_FINE_LEVELS",
    "closed_cavity_and_no_overlap": "CAVITY_GEOMETRY_OR_TOPOLOGY",
    "independent_intact_cut": "CRACK_TOPOLOGY",
    "unique_live_boundary_edge_owners": "CAVITY_EDGE_OWNERSHIP",
    "free_residual_relative": "EQUILIBRIUM",
    "reaction_balance": "GLOBAL_BALANCE",
    "energy_identity": "ENERGY_IDENTITY",
    "reaction_stable_limit": "GLOBAL_LIMIT_STABILITY",
    "compliance_stable_limit": "GLOBAL_LIMIT_STABILITY",
    "energy_stable_limit": "GLOBAL_LIMIT_STABILITY",
    "reaction_fine_accuracy": "GLOBAL_FINAL_LEVEL_ACCURACY",
    "compliance_fine_accuracy": "GLOBAL_FINAL_LEVEL_ACCURACY",
    "energy_fine_accuracy": "GLOBAL_FINAL_LEVEL_ACCURACY",
    "fine_raw_full_boundary_traction": "RAW_CAVITY_TRACTION",
    "fixed_tip_stable_limit": "FIXED_TIP_RECOVERY",
    "fixed_tip_fine_accuracy": "FIXED_TIP_RECOVERY",
    "recovered_fixed_arc_stable_limit": "FIXED_ARC_RECOVERY",
    "recovered_fixed_arc_fine_accuracy": "FIXED_ARC_RECOVERY",
}
SENTINEL_FAMILIES = (
    "centered:{mesh}",
    "offset:-4e-05:{mesh}",
    "offset:4e-05:{mesh}",
    "matrix:4e-05:1.0:{mesh}",
    "far:0.0012:{mesh}",
)


def _first_failure(gates: Mapping[str, bool]) -> str | None:
    for name in FAILURE_ORDER:
        if name in gates and not gates[name]:
            return name
    remaining = sorted(name for name, passed in gates.items() if not passed)
    return remaining[0] if remaining else None


def _sequence(rows: Mapping, ids: list[str], path: tuple[str, ...]):
    values = []
    for source_id in ids:
        value = rows[source_id]
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
            if value is None:
                break
        values.append(deepcopy(value))
    return values


def build_failure_atlas(retained_report: Mapping) -> dict:
    decision = retained_report["decision"]
    rows = retained_report["rows"]
    families = []
    for family in decision["families"]:
        source_ids = list(family["source_ids"])
        configurations = [deepcopy(rows[source_id]["input_configuration"]) for source_id in source_ids]
        levels = [int(configuration["boundary_segments"]) for configuration in configurations]
        qualities = _sequence(rows, source_ids, ("measurements", "mesh_quality"))
        failure = _first_failure(family.get("gates", {}))
        derivative_rows = [deepcopy(row) for row in decision.get("derivatives", ())
                           if family["family"].startswith(row.get("kind", "") + ":")]
        cavity_tensors = []
        for source_id in source_ids:
            recovery = rows[source_id].get("recovery", ())
            cavity_tensors.append([
                {
                    "arc_fraction": item.get("arc_fraction"),
                    "tensor_Pa": deepcopy(item.get("recovery", {}).get("tensor_Pa")),
                    "status": item.get("status", "RECOVERED" if "recovery" in item else "UNAVAILABLE"),
                }
                for item in recovery
            ])
        geometry_fields = (
            "cavity_center_m", "cavity_radius_m", "specimen_width_m", "specimen_height_m",
            "crack_path_m", "crack_enabled", "cavity_enabled", "opening_m",
        )
        geometry = [{key: deepcopy(configuration.get(key)) for key in geometry_fields}
                    for configuration in configurations]
        geometry_match = all(value == geometry[0] for value in geometry[1:])
        families.append({
            "family": family["family"],
            "source_ids": source_ids,
            "mesh_levels": levels,
            "geometry": geometry,
            "geometry_match_across_nominal_refinement": geometry_match,
            "mesh_quality_sequence": qualities,
            "quality_valid_levels": [level for level, quality in zip(levels, qualities)
                                     if quality is not None and quality >= LIMITS["mesh_minimum_quality"]],
            "first_failed_predicate": failure,
            "failure_classification": None if failure is None else FAILURE_CLASS.get(failure, "OTHER_FROZEN_PREDICATE"),
            "all_failed_predicates": [name for name, passed in family.get("gates", {}).items() if not passed],
            "reaction_sequence_N_per_m": _sequence(rows, source_ids, ("measurements", "reaction")),
            "compliance_sequence_m2_per_N": _sequence(rows, source_ids, ("measurements", "compliance")),
            "energy_sequence_J_per_m": _sequence(rows, source_ids, ("measurements", "energy")),
            "raw_normalized_traction_sequence": _sequence(rows, source_ids, ("measurements", "cavity_fields", "normalized_traction")),
            "recovered_fixed_arc_tensor_sequence": cavity_tensors,
            "fixed_tip_tensor_sequence_Pa": _sequence(rows, source_ids, ("measurements", "fixed_tip_probe", "tensor_Pa")),
            "derivative_sequence": derivative_rows,
            "frozen_v1_gates": deepcopy(family.get("gates", {})),
            "frozen_v1_convergence": deepcopy(family.get("convergence", {})),
            "STATIC_FAMILY_V1": "PASS" if family.get("passed") else "FAIL",
        })
    sentinel = [row for row in families if row["family"] in SENTINEL_FAMILIES]
    return {
        "schema": SCHEMA,
        "interpretation": "POSTPROCESSING_RETAINED_V1_SOLVES_NOT_A_V2_PHYSICAL_EXECUTION",
        "STATIC_FAMILY_V1": {
            "passed": sum(row["STATIC_FAMILY_V1"] == "PASS" for row in families),
            "total": len(families),
            "derivatives_passed": sum(bool(row.get("passed")) for row in decision.get("derivatives", ())),
            "derivatives_total": len(decision.get("derivatives", ())),
        },
        "STATIC_FAMILY_V2": "NOT_RUN_SENTINELS_NOT_YET_STABLE",
        "v2_contract": {
            "tolerances": deepcopy(LIMITS),
            "minimum_quality_valid_fine_levels": 2,
            "global_limit_stability_required": True,
            "final_level_global_accuracy_required": True,
            "fixed_tip_operator_required": True,
            "fixed_arc_operator_required_for_cavity_cases": True,
            "derivative_agreement_and_perturbation_convergence_required": True,
            "all_v1_measurements_retained": True,
            "production_source_operator_unchanged": True,
        },
        "sentinel_families": list(SENTINEL_FAMILIES),
        "sentinel_v1_passed": sum(row["STATIC_FAMILY_V1"] == "PASS" for row in sentinel),
        "sentinel_v1_total": len(sentinel),
        "full_v2_matrix_authorized": False,
        "families": families,
    }


__all__ = ["SCHEMA", "SENTINEL_FAMILIES", "build_failure_atlas"]
