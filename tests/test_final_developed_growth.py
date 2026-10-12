from __future__ import annotations

from arrhenius_fracture.fracture_observation_history import ObservationHistory
from scripts.analyze_current_science_pilots_v2 import event_pairs
from scripts.analyze_final_developed_growth import CASES, selected_resistance
from scripts.export_physical_growth_packet import history_rows


def test_accepted_developed_histories_select_exact_event_resistance():
    for label, (case, token) in CASES.items():
        history, steps, _audit, _records = history_rows(label, case, token)
        rows = selected_resistance(label, history, steps)
        assert len(rows) == 203 == len(event_pairs(history))
        assert all(row["event_projected_advance_um"] > 0.0 for row in rows)
        assert all(row["event_path_advance_um"] > 0.0 for row in rows)
        assert {row["load_direction"] for row in rows} == {"loading"}
        assert len({row["load_segment"] for row in rows}) == 1


def test_developed_length_queries_bracket_without_interpolation():
    for _label, (case, _token) in CASES.items():
        history = ObservationHistory.load(case / "fracture_observations_online.jsonl")
        for target_um in (100.0, 500.0, 1000.0):
            query = history.at_length(target_um * 1.0e-6)
            assert query.status == "discrete_jump_bracket"
            assert query.interpolation_performed is False
            assert query.lower is not None and query.upper is not None
            assert query.lower.stage.value == "pre_event"
            assert query.upper.stage.value == "committed"
            assert query.lower.event_id == query.upper.event_id
