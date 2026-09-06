import pytest

from arrhenius_fracture.finalization_v3_schema import (
    PERSISTENT_LIMITATIONS, PROVENANCE_FIELDS, REPRODUCIBILITY_COMPARISON_POLICY,
    SCIENTIFIC_ACCEPTANCE_TOLERANCES, canonical_hash, validate_evidence_rows,
    validate_schema_contract,
)


def payload():
    return {
        "provenance": {key: "0" * 40 for key in PROVENANCE_FIELDS},
        "persistent_limitations": list(PERSISTENT_LIMITATIONS),
        "scientific_tolerances": dict(SCIENTIFIC_ACCEPTANCE_TOLERANCES),
        "reproducibility_policy": dict(REPRODUCIBILITY_COMPARISON_POLICY),
    }


def test_v3_schema_requires_complete_split_provenance_and_persistent_limits():
    assert validate_schema_contract(payload())


def test_v3_schema_rejects_missing_provenance():
    value = payload()
    del value["provenance"]["executed_code_sha"]
    with pytest.raises(ValueError, match="MISSING_V3_PROVENANCE"):
        validate_schema_contract(value)


def test_v3_schema_rejects_removed_persistent_limitation():
    value = payload()
    value["persistent_limitations"].pop()
    with pytest.raises(ValueError, match="PERSISTENT_LIMITATIONS_MISMATCH"):
        validate_schema_contract(value)


def test_v3_evidence_rows_are_recomputed_from_raw_predicate_inputs():
    code = "1" * 40
    configuration = {"partitions": 4}
    geometry = {"stage": "birth_hit_1"}
    row = {
        "case_id": "transition:birth_hit_1:p4", "execution_id": "execution:birth:p4",
        "input_configuration": configuration, "input_hash": canonical_hash(configuration),
        "actual_realized_geometry": geometry, "actual_geometry_fingerprint": canonical_hash(geometry),
        "actual_operation_trace": ["birth_hit_1"], "initial_fingerprint": "a",
        "terminal_fingerprint": "b", "measurement_source": "raw:birth:p4",
        "predicate_name": "transition_partition_exact",
        "predicate_inputs": {"actual_transition_executed": True, "event_identity_equal": True,
          "threshold_rng_equal": True, "complete_terminal_equal": True,
          "event_time_relative_error": 0.0, "event_time_tolerance": 1e-12},
        "predicate_result": True, "source_row_ids": ["raw:birth:p4"],
        "executed_code_sha": code, "claimed_transition": "birth_hit_1",
    }
    assert validate_evidence_rows([row], {"raw:birth:p4": {}}, executed_code_sha=code)
    row["actual_operation_trace"] = ["clock_only"]
    with pytest.raises(ValueError, match="operation_trace"):
        validate_evidence_rows([row], {"raw:birth:p4": {}}, executed_code_sha=code)
