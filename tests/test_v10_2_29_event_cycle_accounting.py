import math

from arrhenius_fracture.persistent_site_cyclic_v10229 import (
    cycle_count_from_consumed_time,
)


def test_consumed_cycles_not_requested_cycles_control_event_time():
    frequency = 1000.0
    requested_cycles = 10000.0
    consumed_time_s = 2.75
    consumed_cycles = cycle_count_from_consumed_time(consumed_time_s, frequency)
    assert consumed_cycles == 2750.0
    assert consumed_cycles < requested_cycles


def test_cycle_count_is_nonnegative():
    assert cycle_count_from_consumed_time(-1.0, 1000.0) == 0.0
    assert cycle_count_from_consumed_time(1.0, -1000.0) == 0.0
    assert math.isclose(cycle_count_from_consumed_time(0.25, 4.0), 1.0)
