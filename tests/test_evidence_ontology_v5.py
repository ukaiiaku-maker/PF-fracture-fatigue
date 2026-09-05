import csv
from pathlib import Path

import pytest

from arrhenius_fracture.evidence_ontology_v5 import canonical_hash, validate_evidence_rows

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/voiding_v5_finalization"


@pytest.mark.parametrize("filename", [
    "end_to_end_matrix.csv", "kinetic_partition_matrix.csv",
    "restart_matrix.csv", "neutrality_matrix.csv", "energy_derivatives.csv",
])
def test_superseded_aliased_tables_are_rejected(filename):
    rows = list(csv.DictReader((OLD / filename).open()))
    with pytest.raises(ValueError, match="MISSING_PROVENANCE"):
        validate_evidence_rows(rows)


def valid_row(case_id="a", execution_id="execution-a"):
    configuration = {"load": 1}
    geometry = {"nodes": [[0, 0]]}
    return {
        "case_id": case_id, "execution_id": execution_id,
        "input_configuration": configuration, "input_hash": canonical_hash(configuration),
        "actual_realized_geometry": geometry, "actual_geometry_fingerprint": canonical_hash(geometry),
        "actual_operation_trace": ["birth"], "initial_fingerprint": "i",
        "terminal_fingerprint": "t", "measurement_source": "runner",
        "predicate_name": "source_boolean", "predicate_inputs": {"source_bindings": {
            "measurement": {"source_row_id": "raw", "path": ["passed"]}
        }},
        "predicate_result": True, "source_row_ids": ["raw"], "implementation_sha": "sha",
    }


def validate(rows):
    return validate_evidence_rows(rows, source_rows={"raw": {"passed": True}}, implementation_sha="sha")


def test_distinct_claims_cannot_share_one_execution():
    base = valid_row()
    with pytest.raises(ValueError, match="ALIASED_VARIANT_EXECUTION"):
        validate(({**base, "case_id": "a"}, {**base, "case_id": "b", "execution_id": "execution-b"}))


@pytest.mark.parametrize("mutation,code", [
    ({"input_hash": "fabricated"}, "INPUT_HASH_MISMATCH"),
    ({"actual_geometry_fingerprint": "fabricated"}, "GEOMETRY_FINGERPRINT_MISMATCH"),
    ({"implementation_sha": "wrong"}, "IMPLEMENTATION_SHA_MISMATCH"),
    ({"source_row_ids": ["absent"]}, "UNTRACEABLE_MEASUREMENT"),
    ({"predicate_result": False}, "PREDICATE_RESULT_MISMATCH"),
])
def test_full_schema_fabrication_is_rejected(mutation, code):
    with pytest.raises(ValueError, match=code):
        validate(({**valid_row(), **mutation},))


def test_one_execution_cannot_be_relabelled_with_distinct_inputs():
    first = valid_row("a", "same-execution")
    configuration = {"load": 2}
    second = {**valid_row("b", "same-execution"), "input_configuration": configuration,
              "input_hash": canonical_hash(configuration), "terminal_fingerprint": "t2"}
    with pytest.raises(ValueError, match="EXECUTION_RELABELED_WITH_DIFFERENT_INPUTS"):
        validate((first, second))


def test_expected_equivalence_requires_independent_executions():
    base = {**valid_row(), "equality_classification": "EXPECTED_EQUIVALENCE"}
    with pytest.raises(ValueError, match="EXPECTED_EQUIVALENCE_NOT_INDEPENDENT"):
        validate((base, {**base, "case_id": "b"}))


@pytest.mark.parametrize("filename", [
    "end_to_end_matrix.csv", "kinetic_partition_matrix.csv",
    "restart_matrix.csv", "neutrality_matrix.csv", "energy_derivatives.csv",
])
def test_superseded_tables_remain_rejected_after_superficial_schema_backfill(filename):
    legacy = list(csv.DictReader((OLD / filename).open()))
    assert len(legacy) >= 2
    first = valid_row("legacy-a", "reused-legacy-execution")
    second_configuration = {"legacy_row": legacy[1]}
    second = {
        **valid_row("legacy-b", "reused-legacy-execution"),
        "input_configuration": second_configuration,
        "input_hash": canonical_hash(second_configuration),
        "terminal_fingerprint": "legacy-terminal-b",
    }
    with pytest.raises(ValueError, match="EXECUTION_RELABELED_WITH_DIFFERENT_INPUTS"):
        validate((first, second))
