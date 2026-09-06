"""Prospectively frozen schema and scientific criteria for finalization-v3."""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "v12.voiding-v5-finalization-v3/2"
SHA40 = re.compile(r"^[0-9a-f]{40}$")

PERSISTENT_LIMITATIONS = (
    "single_void_only", "two_dimensional_plane_strain", "uncalibrated_parameters",
    "not_experimentally_validated", "no_general_multi_front_campaign",
    "no_fatigue_calibration", "no_fatigue_growth_campaign",
    "standard_absolute_K_unavailable",
)

# evidence_commit_sha is intentionally external: an artifact cannot contain the
# hash of the future commit containing itself.
PROVENANCE_FIELDS = (
    "model_form_base_sha", "stochastic_seed_policy_sha", "criteria_sha",
    "implementation_sha", "executed_code_sha", "campaign_runner_sha",
    "static_qualifier_sha", "production_qualifier_sha", "evidence_generation_sha",
)

EVIDENCE_FIELDS = (
    "case_id", "execution_id", "input_configuration", "input_hash",
    "actual_realized_geometry", "actual_geometry_fingerprint", "actual_operation_trace",
    "initial_fingerprint", "terminal_fingerprint", "measurement_source",
    "predicate_name", "predicate_inputs", "predicate_result", "source_row_ids",
    "executed_code_sha",
)

SCIENTIFIC_ACCEPTANCE_TOLERANCES = {
    "requested_realized_geometry_abs_m": 1.0e-12,
    "inventory_identity_abs_m2": 1.0e-24,
    "length_identity_abs_m": 1.0e-15,
    "transition_event_time_relative": 1.0e-12,
    "free_residual_relative": 1.0e-8,
    "reaction_balance_relative": 3.0e-2,
    "energy_reaction_identity_relative": 1.0e-2,
    "cavity_area_relative": 2.0e-2,
    "cavity_perimeter_relative": 2.0e-2,
    "kirsch_relative": 3.0e-2,
    "cavity_traction_normalized": 5.0e-2,
    "mesh_minimum_quality": 5.0e-2,
    "far_void_relative": 2.0e-2,
    "tensor_probe_relative": 5.0e-2,
    "derivative_perturbation_relative": 1.0e-1,
    "derivative_energy_compliance_relative": 1.0e-1,
    "static_mesh_reaction_relative": 6.0e-2,
    "static_mesh_energy_relative": 6.0e-2,
    "offset_reaction_relative": 5.0e-3,
    "offset_compliance_relative": 5.0e-3,
}

REPRODUCIBILITY_COMPARISON_POLICY = {
    "diagnostic_float_relative": 1.0e-12,
    "diagnostic_float_absolute": 1.0e-15,
    "exact_classes": (
        "gate_identity", "threshold", "event_id", "source_relationship", "input_hash",
        "topology_fingerprint", "support_fingerprint", "operation_trace", "categorical_outcome",
    ),
}

FROZEN_CASE_REGISTRY = {
    "transitions": ("birth_hit_1", "birth_hit_2", "stabilization", "healing", "subgrid_growth",
                    "promotion", "ligament", "downstream_child", "child_continuation"),
    "restarts": ("available_site", "incomplete_first_hit", "between_birth_hits", "embryo",
                 "stable_subgrid_cavity", "before_promotion", "after_promotion", "before_ligament",
                 "connected_before_downstream", "downstream_child_before_continuation",
                 "zero_drive_connected"),
    "controlled": ("centered", "positive_offset", "negative_offset", "short_ligament",
                   "long_ligament", "diffusion_limited", "accommodation_limited", "embryo_healing",
                   "downstream_zero_drive", "delayed_downstream", "fixed_mesh_oblique",
                   "local_remesh_refinement"),
    "neutrality": ("monotonic", "fixed_mesh_oblique", "checkpoint_restart", "unload_reload"),
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _all_within(v):
    return all(abs(float(row["error"])) <= float(row["tolerance"]) for row in v["comparisons"])


REGISTERED_SCIENTIFIC_PREDICATES = {
    "fixed_geometry_exact": lambda v: v["maximum_error_m"] <= v["tolerance_m"] and v["ray_intersects_polygon"],
    "equilibrium": lambda v: v["free_residual_relative"] <= v["free_residual_tolerance"] and v["reaction_balance"] <= v["reaction_tolerance"] and v["energy_reaction_identity"] <= v["energy_tolerance"],
    "mesh_and_cavity_topology": lambda v: v["closed_cycle"] and v["solid_inside_count"] == 0 and v["wake_overlap_count"] == 0 and not v["bridge_path"] and v["support_certified"] and v["minimum_quality"] >= v["quality_minimum"],
    "bounded_convergence": _all_within,
    "transition_partition_exact": lambda v: v["actual_transition_executed"] and v["event_identity_equal"] and v["threshold_rng_equal"] and v["complete_terminal_equal"] and v["event_time_relative_error"] <= v["event_time_tolerance"],
    "restart_common_terminal_exact": lambda v: v["checkpoint_stage"] == v["restart_stage"] and v["continued_to_common_terminal"] and v["events_equal"] and v["terminal_equal"],
    "rollback_exact": lambda v: v["failure_injected"] and v["exception_observed"] and v["restored_exactly"],
    "v12_disabled_neutrality_exact": lambda v: v["base_model"] == v["disabled_model"] and v["complete_trajectory_equal"],
    "stagewise_conservation": _all_within,
    "natural_reproducibility": lambda v: v["executed"] and v["partition_equal"] and v["restart_equal"] and v["rng_equal"],
    "reaction_balance": lambda v: abs(v["imbalance"]) <= v["tolerance"],
    "energy_reaction_identity": lambda v: abs(v["relative_error"]) <= v["tolerance"],
    "cavity_area_convergence": _all_within,
    "cavity_perimeter_convergence": _all_within,
    "kirsch_convergence": _all_within,
    "cavity_traction": lambda v: v["normalized_traction"] <= v["tolerance"],
    "mesh_quality": lambda v: v["minimum_quality"] >= v["minimum_allowed"],
    "no_solid_inside_cavity": lambda v: v["intersecting_triangle_count"] == 0,
    "no_wake_cavity_overlap": lambda v: v["overlap_count"] == 0,
    "no_intact_bridge_path": lambda v: not v["bridge_path_exists"],
    "closed_cavity_topology": lambda v: v["component_count"] == 1 and v["all_degrees_two"],
    "tensor_probe_convergence": _all_within,
    "far_void_convergence": _all_within,
    "fixed_crack_offset_symmetry": _all_within,
    "crack_derivative_agreement": _all_within,
    "cavity_derivative_agreement": _all_within,
    "perturbation_size_convergence": _all_within,
    "mesh_convergence": _all_within,
    "natural_seed_execution": lambda v: v["executed"] and v["terminal_classification_declared"],
    "natural_partition_invariance": lambda v: v["partition_equal"],
    "natural_restart_invariance": lambda v: v["restart_equal"],
    "natural_rng_reproducibility": lambda v: v["rng_equal"],
}


def validate_evidence_rows(rows, source_rows, *, executed_code_sha):
    rows = tuple(dict(row) for row in rows)
    errors = []
    seen_cases, seen_executions = set(), set()
    for row in rows:
        missing = [key for key in EVIDENCE_FIELDS if key not in row]
        if missing:
            errors.append((row.get("case_id"), "missing", missing)); continue
        if row["case_id"] in seen_cases or row["execution_id"] in seen_executions:
            errors.append((row["case_id"], "aliased_execution"))
        seen_cases.add(row["case_id"]); seen_executions.add(row["execution_id"])
        if canonical_hash(row["input_configuration"]) != row["input_hash"]:
            errors.append((row["case_id"], "input_hash"))
        if canonical_hash(row["actual_realized_geometry"]) != row["actual_geometry_fingerprint"]:
            errors.append((row["case_id"], "geometry_hash"))
        if row["executed_code_sha"] != executed_code_sha or not SHA40.fullmatch(row["executed_code_sha"]):
            errors.append((row["case_id"], "executed_code_sha"))
        if not row["source_row_ids"] or any(key not in source_rows for key in row["source_row_ids"]):
            errors.append((row["case_id"], "sources"))
        predicate = REGISTERED_SCIENTIFIC_PREDICATES.get(row["predicate_name"])
        if predicate is None or bool(predicate(row["predicate_inputs"])) != bool(row["predicate_result"]):
            errors.append((row["case_id"], "predicate"))
        claimed = row.get("claimed_transition")
        if claimed and claimed not in row["actual_operation_trace"]:
            errors.append((row["case_id"], "operation_trace"))
    if errors:
        raise ValueError(errors)
    return rows


def validate_schema_contract(payload):
    provenance = payload.get("provenance", {})
    missing = [field for field in PROVENANCE_FIELDS if field not in provenance]
    if missing:
        raise ValueError({"code": "MISSING_V3_PROVENANCE", "fields": missing})
    if any(not SHA40.fullmatch(str(provenance[field])) for field in PROVENANCE_FIELDS):
        raise ValueError({"code": "INVALID_V3_PROVENANCE_SHA"})
    if tuple(payload.get("persistent_limitations", ())) != PERSISTENT_LIMITATIONS:
        raise ValueError({"code": "PERSISTENT_LIMITATIONS_MISMATCH"})
    if payload.get("scientific_tolerances") != SCIENTIFIC_ACCEPTANCE_TOLERANCES:
        raise ValueError({"code": "SCIENTIFIC_TOLERANCES_MISMATCH"})
    if payload.get("reproducibility_policy") != REPRODUCIBILITY_COMPARISON_POLICY:
        raise ValueError({"code": "REPRODUCIBILITY_POLICY_MISMATCH"})
    return payload


__all__ = [name for name in globals() if name.isupper()] + ["canonical_hash", "validate_evidence_rows", "validate_schema_contract"]
