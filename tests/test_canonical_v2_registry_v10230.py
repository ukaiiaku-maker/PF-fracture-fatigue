import pytest

from arrhenius_fracture.canonical_v2_registry_v10230 import (
    decoded_contract, load_row, load_rows, round_trip,
)


def test_eight_sealed_rows_round_trip_byte_fields_exactly():
    rows = load_rows()
    assert len(rows) == 8
    for row in rows.values():
        assert round_trip(row) == row
        contract = decoded_contract(row)
        assert contract["complete_bound_row_sha256"] == row["complete_bound_row_sha256"]
        assert contract["cleavage_hits"] == float(row["physics__cleavage_hits"])
        assert contract["cleavage_correlation_time_s"] == float(
            row["physics__cleavage_correlation_time_s"]
        )


def test_unknown_alias_fails_closed():
    with pytest.raises(KeyError, match="unknown canonical candidate alias"):
        load_row("not-a-canonical-row")
