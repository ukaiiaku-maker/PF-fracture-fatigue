import pytest

from arrhenius_fracture.canonical_v2_registry_v10230 import (
    decoded_contract, load_row, load_rows, round_trip, surface_adapters,
)
from scripts.run_v2_2_1d_rising_resistance_v10230 import merge_event_and_state


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


def test_analytical_monotonic_fatigue_pf_and_fem_surface_interfaces_are_identical():
    stress = [0.0, 1.0e9, 5.0e9, 30.0e9]
    for row in load_rows().values():
        for adapter in surface_adapters(row):
            assert adapter.parity(stress, 300.0)
            assert adapter.parity(stress, 900.0)
            assert (adapter.rate(stress, 900.0) >= 0.0).all()


def test_fixed_event_threshold_is_not_shadowed_by_post_event_engine_snapshot():
    merged = merge_event_and_state(
        {"threshold_action": 0.433209, "physical_hazard_action": 0.0},
        {"threshold_action": 0.6931471805599453, "physical_hazard_action": 0.6931471805599453},
    )
    assert merged["threshold_action"] == 0.6931471805599453
    assert merged["physical_hazard_action"] == 0.6931471805599453
