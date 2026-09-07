import pytest
from copy import deepcopy

from arrhenius_fracture.finalization_v3_closure_schema import (
    CONTROLLED_TERMINAL_CLASSIFICATIONS, CONTROLLED_TRACE_TOKENS, PARTITIONS,
    PHYSICAL_INPUT_FIELDS, TRANSITION_TRACE_TOKENS,
    expected_registry_keys, validate_closure_evidence,
)
from arrhenius_fracture.finalization_v3_schema import FROZEN_CASE_REGISTRY, canonical_hash


def test_current_lifecycle_schema_routes_to_strict_lifecycle_validator(monkeypatch):
    from arrhenius_fracture import closure_lifecycle_evidence as lifecycle
    seen=[]
    def validate(payload,sources,*,executed_code_sha):
        seen.append((payload,sources,executed_code_sha)); return 'lifecycle'
    monkeypatch.setattr(lifecycle,'validate_lifecycle',validate)
    payload={'schema':lifecycle.SCHEMA}; sources={}
    assert validate_closure_evidence(payload,sources,executed_code_sha='1'*40)=='lifecycle'
    assert seen==[(payload,sources,'1'*40)]


def test_unknown_lifecycle_version_remains_rejected_by_strict_validator():
    with pytest.raises(ValueError,match='lifecycle source/schema identity'):
        validate_closure_evidence({'schema':'v12.voiding-v5-closure-actual-lifecycle/999'}, {},
                                  executed_code_sha='1'*40)


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
    trace = list(TRANSITION_TRACE_TOKENS.get(case, (case,))) + [case]
    return {"case_id": f"{case}:p{partition}", "execution_id": f"exec:{case}:p{partition}",
            "input_configuration": configuration, "input_hash": canonical_hash(configuration),
            "initial_state": initial, "initial_fingerprint": canonical_hash(initial),
            "terminal_state": terminal, "terminal_fingerprint": canonical_hash(terminal),
            "actual_realized_geometry": {"generation": 0},
            "actual_geometry_fingerprint": canonical_hash({"generation": 0}),
            "actual_operation_trace": trace, "claimed_transition": case,
            "predicate_name": "bounded_convergence",
            "predicate_inputs": {"comparisons": [{"error": 0.0, "tolerance": 0.0}]},
            "predicate_result": True, "source_row_ids": [f"source:{case}:p{partition}"],
            "measurement_source": "accepted_production_state_capture",
            "observed_terminal_classification": configuration["expected_terminal_classification"],
            "executed_code_sha": code}


def complete_rows():
    rows = [row(case, partition) for case in FROZEN_CASE_REGISTRY["transitions"] for partition in PARTITIONS]
    for dataset in ("restarts", "controlled", "neutrality"):
        for case in FROZEN_CASE_REGISTRY[dataset]:
            value = row(case, None); value["input_configuration"]["dataset"] = dataset
            value["case_id"] = f"{dataset}:{case}"
            value["execution_id"] = f"exec:{dataset}:{case}"
            value["source_row_ids"] = [f"source:{dataset}:{case}"]
            value["input_hash"] = canonical_hash(value["input_configuration"])
            if dataset not in {"controlled"}:
                value.pop("claimed_transition"); value["input_configuration"]["transition_identity"] = None
                value["input_hash"] = canonical_hash(value["input_configuration"])
            else:
                value["actual_operation_trace"] = list(CONTROLLED_TRACE_TOKENS[case]) + [case]
                terminal = CONTROLLED_TERMINAL_CLASSIFICATIONS[case][0]
                value["input_configuration"]["expected_terminal_classification"] = terminal
                value["observed_terminal_classification"] = terminal
                value["input_hash"] = canonical_hash(value["input_configuration"])
            rows.append(value)
    return rows


def sources_for(rows):
    return {item["source_row_ids"][0]: {
        "fingerprint_method": "complete_accepted_state_fingerprint",
        "initial_accepted_state_fingerprint": item["initial_fingerprint"],
        "terminal_accepted_state_fingerprint": item["terminal_fingerprint"],
    } for item in rows}


def test_complete_physical_registry_validates():
    rows = complete_rows(); sources = sources_for(rows)
    assert len(validate_closure_evidence(rows, sources, executed_code_sha="1" * 40)) == len(rows)


def test_generic_trace_and_configuration_fingerprint_are_rejected():
    rows = complete_rows(); sources = sources_for(rows)
    rows[0]["actual_operation_trace"] = ["transition_partitions"]
    rows[0]["initial_fingerprint"] = rows[0]["input_hash"]
    with pytest.raises(ValueError, match="operation_trace"):
        validate_closure_evidence(rows, sources, executed_code_sha="1" * 40)


def test_omitted_second_birth_and_zero_drive_restart_are_rejected():
    rows = [item for item in complete_rows()
            if item["input_configuration"]["case_identity"] not in {"birth_hit_2", "zero_drive_connected"}]
    sources = sources_for(rows)
    with pytest.raises(ValueError, match="registry"):
        validate_closure_evidence(rows, sources, executed_code_sha="1" * 40)


@pytest.mark.parametrize("field", ["case_id", "execution_id"])
def test_duplicate_identifiers_are_rejected(field):
    rows = complete_rows(); rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match="aliased_execution"):
        validate_closure_evidence(rows, sources_for(rows), executed_code_sha="1" * 40)


def test_fabricated_shared_base_and_arbitrary_state_binding_are_rejected():
    rows = complete_rows()
    left, right = rows[0], rows[1]
    right["input_configuration"] = deepcopy(left["input_configuration"])
    right["input_hash"] = left["input_hash"]
    left["shared_base_execution_id"] = right["shared_base_execution_id"] = "does-not-exist"
    left["derived_predicate"] = right["derived_predicate"] = True
    sources = sources_for(rows)
    sources[left["source_row_ids"][0]]["initial_accepted_state_fingerprint"] = "fabricated"
    with pytest.raises(ValueError, match="accepted_state_source_binding|aliased_physical_input"):
        validate_closure_evidence(rows, sources, executed_code_sha="1" * 40)


def test_missing_required_remesh_and_terminal_mismatch_are_rejected():
    rows = complete_rows(); promotion = next(
        item for item in rows if item["input_configuration"]["case_identity"] == "promotion")
    promotion["actual_operation_trace"].remove("remesh")
    promotion["observed_terminal_classification"] = "wrong"
    with pytest.raises(ValueError, match="operation_trace"):
        validate_closure_evidence(rows, sources_for(rows), executed_code_sha="1" * 40)


def test_reversed_and_duplicated_required_operations_are_rejected():
    rows = complete_rows(); promotion = next(item for item in rows
        if item["input_configuration"]["case_identity"] == "promotion")
    promotion["actual_operation_trace"] = list(reversed(promotion["actual_operation_trace"]))
    with pytest.raises(ValueError, match="operation_trace"):
        validate_closure_evidence(rows, sources_for(rows), executed_code_sha="1" * 40)
    rows = complete_rows(); promotion = next(item for item in rows
        if item["input_configuration"]["case_identity"] == "promotion")
    promotion["actual_operation_trace"].insert(1, "remesh")
    with pytest.raises(ValueError, match="operation_trace"):
        validate_closure_evidence(rows, sources_for(rows), executed_code_sha="1" * 40)


def test_runner_authored_but_unfrozen_terminal_is_rejected():
    rows = complete_rows(); centered = next(item for item in rows
        if item["input_configuration"]["case_identity"] == "centered")
    centered["input_configuration"]["expected_terminal_classification"] = "OPPORTUNISTIC_PASS"
    centered["observed_terminal_classification"] = "OPPORTUNISTIC_PASS"
    centered["input_hash"] = canonical_hash(centered["input_configuration"])
    with pytest.raises(ValueError, match="frozen_terminal_classification"):
        validate_closure_evidence(rows, sources_for(rows), executed_code_sha="1" * 40)
