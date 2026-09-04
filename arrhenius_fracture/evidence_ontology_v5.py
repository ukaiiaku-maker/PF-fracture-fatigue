"""Fail-closed provenance validation for V5 scientific evidence rows."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Any

REQUIRED = (
    "case_id", "input_configuration", "input_hash", "actual_realized_geometry",
    "actual_operation_trace", "initial_fingerprint", "terminal_fingerprint",
    "measurement_source", "predicate_name", "predicate_inputs",
    "predicate_result", "source_row_ids", "implementation_sha",
)


def validate_evidence_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict, ...]:
    values = tuple(dict(row) for row in rows)
    errors = []
    ids = [row.get("case_id") for row in values]
    if len(ids) != len(set(ids)):
        errors.append({"code": "DUPLICATE_CASE_ID"})
    signatures = defaultdict(list)
    for row in values:
        missing = [field for field in REQUIRED if field not in row]
        if missing:
            errors.append({"case_id": row.get("case_id"), "code": "MISSING_PROVENANCE", "fields": missing})
            continue
        if not row["predicate_name"] or not isinstance(row["predicate_inputs"], Mapping):
            errors.append({"case_id": row["case_id"], "code": "LITERAL_OR_UNNAMED_PREDICATE"})
        if not row["measurement_source"] or not row["source_row_ids"]:
            errors.append({"case_id": row["case_id"], "code": "UNTRACEABLE_MEASUREMENT"})
        claimed = row.get("claimed_transition")
        if claimed and claimed not in row["actual_operation_trace"]:
            errors.append({"case_id": row["case_id"], "code": "TRANSITION_NOT_EXECUTED", "transition": claimed})
        stage = row.get("restart_stage")
        if stage and row.get("checkpoint_stage") != stage:
            errors.append({"case_id": row["case_id"], "code": "RESTART_STAGE_ALIASED"})
        if row.get("evidence_type") == "neutrality" and len(row.get("independent_peer_execution_ids", ())) != 2:
            errors.append({"case_id": row["case_id"], "code": "NEUTRALITY_PEERS_NOT_EXECUTED"})
        if row.get("evidence_type") == "derivative" and len(row["source_row_ids"]) < 2:
            errors.append({"case_id": row["case_id"], "code": "DERIVATIVE_WITHOUT_SOURCE_ROWS"})
        signature = (row["input_hash"], row["terminal_fingerprint"])
        signatures[signature].append(row)
    for group in signatures.values():
        if len(group) > 1 and not all(row.get("equality_classification") == "EXPECTED_EQUIVALENCE" for row in group):
            errors.append({"case_ids": [row["case_id"] for row in group], "code": "ALIASED_VARIANT_EXECUTION"})
    if errors:
        raise ValueError(errors)
    return values


__all__ = ["REQUIRED", "validate_evidence_rows"]
