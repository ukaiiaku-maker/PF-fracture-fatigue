import math

from scripts.analyze_r10_weakt_active_acquisition import developed_window


def test_developed_window_uses_first_events_crossing_requested_endpoints():
    events = [
        {"event_index": 1, "extension_m": 9e-6, "cycles": 10.0},
        {"event_index": 2, "extension_m": 21e-6, "cycles": 30.0},
        {"event_index": 3, "extension_m": 59e-6, "cycles": 70.0},
        {"event_index": 4, "extension_m": 63e-6, "cycles": 80.0},
    ]
    result = developed_window(events)
    assert result["left_event_index"] == 2
    assert result["right_event_index"] == 4
    assert math.isclose(result["rate_m_per_cycle"], 42e-6 / 50.0)
