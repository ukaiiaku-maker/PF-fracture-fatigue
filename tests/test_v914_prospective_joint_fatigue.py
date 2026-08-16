import numpy as np

from scripts.analyze_v914_prospective_joint_fatigue import (
    classify_status,
    developed_from_events,
)


def test_status_semantics_do_not_merge_censor_and_partial():
    assert classify_status("maximum_cycles_reached", 10)[0] == "cycle_or_hazard_censor"
    assert classify_status("periodic_state_failed", 10)[0] == "partial_or_numerical_unresolved"
    assert classify_status("growth_target_reached", 100)[0] == "developed_target_reached"


def test_developed_event_rate_uses_only_post_development_events():
    events = [
        {"cycles": 1.0, "cumulative_extension_m": 10e-6},
        {"cycles": 3.0, "cumulative_extension_m": 25e-6},
        {"cycles": 8.0, "cumulative_extension_m": 50e-6},
    ]
    assert np.isclose(developed_from_events(events), 25e-6 / 5.0)
