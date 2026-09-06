"""Prospectively frozen schema and criteria for V5 finalization-v3."""
from __future__ import annotations

SCHEMA = "v12.voiding-v5-finalization-v3/1"

PERSISTENT_LIMITATIONS = (
    "single_void_only",
    "two_dimensional_plane_strain",
    "uncalibrated_parameters",
    "not_experimentally_validated",
    "no_general_multi_front_campaign",
    "no_fatigue_calibration",
    "no_fatigue_growth_campaign",
    "standard_absolute_K_unavailable",
)

PROVENANCE_FIELDS = (
    "model_form_base_sha",
    "stochastic_seed_policy_sha",
    "executed_code_sha",
    "campaign_runner_sha",
    "static_qualifier_sha",
    "production_qualifier_sha",
    "evidence_generation_sha",
    "evidence_commit_sha",
)

EVIDENCE_FIELDS = (
    "case_id",
    "execution_id",
    "input_configuration",
    "input_hash",
    "actual_realized_geometry",
    "actual_geometry_fingerprint",
    "actual_operation_trace",
    "initial_fingerprint",
    "terminal_fingerprint",
    "measurement_source",
    "predicate_name",
    "predicate_inputs",
    "predicate_result",
    "source_row_ids",
    "executed_code_sha",
)

# These tolerances are frozen before any v3 evidence is generated. Identity,
# categorical, event, threshold, RNG, topology, support, and provenance fields
# are exact and are never evaluated with these numerical tolerances.
TOLERANCES = {
    "requested_realized_geometry_abs_m": 1.0e-12,
    "inventory_identity_abs_m2": 1.0e-24,
    "length_identity_abs_m": 1.0e-15,
    "free_residual_relative": 1.0e-8,
    "reaction_balance_relative": 3.0e-2,
    "energy_reaction_identity_relative": 1.0e-2,
    "cross_worker_float_relative": 1.0e-12,
    "cross_worker_float_absolute": 1.0e-15,
    "static_mesh_reaction_relative": 6.0e-2,
    "static_mesh_energy_relative": 6.0e-2,
    "offset_reaction_relative": 5.0e-3,
    "offset_compliance_relative": 5.0e-3,
}


def validate_schema_contract(payload):
    missing = [field for field in PROVENANCE_FIELDS if field not in payload["provenance"]]
    if missing:
        raise ValueError({"code": "MISSING_V3_PROVENANCE", "fields": missing})
    if tuple(payload.get("persistent_limitations", ())) != PERSISTENT_LIMITATIONS:
        raise ValueError({"code": "PERSISTENT_LIMITATIONS_MISMATCH"})
    if payload.get("tolerances") != TOLERANCES:
        raise ValueError({"code": "PROSPECTIVE_TOLERANCES_MISMATCH"})
    return payload


__all__ = [
    "EVIDENCE_FIELDS", "PERSISTENT_LIMITATIONS", "PROVENANCE_FIELDS",
    "SCHEMA", "TOLERANCES", "validate_schema_contract",
]
