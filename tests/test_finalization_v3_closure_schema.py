import pytest

from arrhenius_fracture.finalization_v3_closure_schema import (
    PARTITIONS, PHYSICAL_INPUT_FIELDS, expected_registry_keys, validate_closure_evidence,
)
from arrhenius_fracture.finalization_v3_schema import FROZEN_CASE_REGISTRY, canonical_hash


def test_registry_requires_45_transition_and_11_restart_cases():
    expected = expected_registry_keys()
    assert len(expected["transitions"]) == 45
    assert len(expected["restarts"]) == 11
    assert ("birth_hit_1", 1) in expected["transitions"]
    assert ("birth_hit_2", 16) in expected["transitions"]
    assert ("zero_drive_connected", None) in expected["restarts"]


def row(case, partition, code="1" * 40):
    configuration = {field: None for field in PHYSICAL_INPUT_FIELDS}
    configuration.update(dataset="transitions", case_identity=case,
                         transition_identity=case, partition_count=partition,
                         initial_stage="before_" + case,
                         expected_terminal_classification="after_" + case,
                         physical_geometry={"generation": 0}, loading_history=[],
                         candidate_event_identity={"event": case})
    initial = {"stage": configuration["initial_stage"], "case": case}
    terminal = {"stage": configuration["expected_terminal_classification"], "case": case}
    return {"case_id": f"{case}:p{partition}", "execution_id": f"exec:{case}:p{partition}",
            "input_configuration": configuration, "input_hash": canonical_hash(configuration),
            "initial_state": initial, "initial_fingerprint": canonical_hash(initial),
            "terminal_state": terminal, "terminal_fingerprint": canonical_hash(terminal),
            "actual_operation_trace": [case], "claimed_transition": case,
            "predicate_name": "bounded_convergence",
            "predicate_inputs": {"comparisons": [{"error": 0.0, "tolerance": 0.0}]},
            "predicate_result": True, "source_row_ids": [f"source:{case}:p{partition}"],
            "executed_code_sha": code}


def complete_rows():
    rows = [row(case, partition) for case in FROZEN_CASE_REGISTRY["transitions"] for partition in PARTITIONS]
    for dataset in ("restarts", "controlled", "neutrality"):
        for case in FROZEN_CASE_REGISTRY[dataset]:
            value = row(case, None); value["input_configuration"]["dataset"] = dataset
            value["input_hash"] = canonical_hash(value["input_configuration"])
            if dataset not in {"controlled"}:
                value.pop("claimed_transition"); value["input_configuration"]["transition_identity"] = None
                value["input_hash"] = canonical_hash(value["input_configuration"])
            rows.append(value)
    return rows


def test_complete_physical_registry_validates():
    rows = complete_rows(); sources = {source: {} for item in rows for source in item["source_row_ids"]}
    assert len(validate_closure_evidence(rows, sources, executed_code_sha="1" * 40)) == len(rows)


def test_generic_trace_and_configuration_fingerprint_are_rejected():
    rows = complete_rows(); sources = {source: {} for item in rows for source in item["source_row_ids"]}
    rows[0]["actual_operation_trace"] = ["transition_partitions"]
    rows[0]["initial_fingerprint"] = rows[0]["input_hash"]
    with pytest.raises(ValueError, match="operation_trace"):
        validate_closure_evidence(rows, sources, executed_code_sha="1" * 40)


def test_omitted_second_birth_and_zero_drive_restart_are_rejected():
    rows = [item for item in complete_rows()
            if item["input_configuration"]["case_identity"] not in {"birth_hit_2", "zero_drive_connected"}]
    sources = {source: {} for item in rows for source in item["source_row_ids"]}
    with pytest.raises(ValueError, match="registry"):
        validate_closure_evidence(rows, sources, executed_code_sha="1" * 40)
