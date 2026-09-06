import pytest

from arrhenius_fracture.finalization_v3_schema import (
    PERSISTENT_LIMITATIONS, PROVENANCE_FIELDS, TOLERANCES, validate_schema_contract,
)


def payload():
    return {
        "provenance": {key: "0" * 40 for key in PROVENANCE_FIELDS},
        "persistent_limitations": list(PERSISTENT_LIMITATIONS),
        "tolerances": dict(TOLERANCES),
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
