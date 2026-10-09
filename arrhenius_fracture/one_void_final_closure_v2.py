"""Native schema and fail-closed validator for final one-void closure V2."""
from __future__ import annotations

from collections.abc import Mapping

SCHEMA = "v5.one-void-final-scientific-closure/2"
TERMINAL = {
    "V5_ONE_VOID_SCIENTIFIC_CLOSURE_QUALIFIED",
    "V5_ONE_VOID_FINAL_SCIENTIFIC_CLOSURE_COMPLETE_BUT_BLOCKED",
}
REQUIRED_GATES = (
    "r_tip_final_classification",
    "static_family_v1_preserved",
    "static_family_v2_final",
    "controlled_histories_final",
    "transition_partitions",
    "common_terminal_restarts",
    "lifecycle_rollback",
    "same_worker_natural_peers",
    "natural_bitwise_replay_v1_preserved",
    "natural_physical_replay_v2",
    "future_causal_disabled_neutrality",
    "stagewise_conservation",
    "paired_evidence_exact",
)


def validate(record: Mapping) -> dict:
    errors = []
    if record.get("schema") != SCHEMA:
        errors.append("schema")
    if record.get("terminal_classification") not in TERMINAL:
        errors.append("terminal_classification")
    gates = record.get("gates", {})
    for name in REQUIRED_GATES:
        if name not in gates:
            errors.append("missing_gate:" + name)
        elif gates[name].get("status") not in ("PASS", "FAIL", "BLOCKED", "NOT_DEFINED"):
            errors.append("invalid_gate_status:" + name)
    ontology = record.get("source_bound_ontology", {})
    rows = ontology.get("rows", ())
    if not rows or ontology.get("row_count") != len(rows):
        errors.append("source_bound_ontology")
    identities = [(row.get("dataset"), row.get("case_identity")) for row in rows]
    if len(identities) != len(set(identities)):
        errors.append("aliased_ontology_identity")
    if any(row.get("classification") not in ("PASS", "BLOCKED") for row in rows):
        errors.append("ontology_classification")
    qualified = record.get("terminal_classification") == "V5_ONE_VOID_SCIENTIFIC_CLOSURE_QUALIFIED"
    mandatory_pass = all(gates.get(name, {}).get("status") == "PASS" for name in REQUIRED_GATES)
    if qualified != mandatory_pass:
        errors.append("terminal_gate_consistency")
    if record.get("r_tip_distinct_from_R_void") is not True:
        errors.append("r_tip_distinct_from_R_void")
    return {"schema": "v5.one-void-final-scientific-closure-validation/2",
            "valid": not errors, "errors": errors}


__all__ = ["REQUIRED_GATES", "SCHEMA", "TERMINAL", "validate"]
