import csv
from pathlib import Path

import pytest

from arrhenius_fracture.evidence_ontology_v5 import validate_evidence_rows

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


def test_distinct_claims_cannot_share_one_execution():
    base = {
        "input_configuration": {}, "input_hash": "same", "actual_realized_geometry": {},
        "actual_operation_trace": ["birth"], "initial_fingerprint": "i",
        "terminal_fingerprint": "t", "measurement_source": "runner",
        "predicate_name": "event_seen", "predicate_inputs": {"event": "birth"},
        "predicate_result": True, "source_row_ids": ["raw"], "implementation_sha": "sha",
    }
    with pytest.raises(ValueError, match="ALIASED_VARIANT_EXECUTION"):
        validate_evidence_rows(({**base, "case_id": "a"}, {**base, "case_id": "b"}))
