import inspect

from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_event_aware_coupled_interval,
)


def test_event_aware_ramp_is_explicit_and_hold_compatible():
    signature = inspect.signature(advance_event_aware_coupled_interval)
    rate = signature.parameters["opening_rate_m_per_s"]
    assert rate.default == 0.0
    source = inspect.getsource(advance_event_aware_coupled_interval)
    assert '"linear_total_opening_ramp"' not in source
    assert '"ramp" if opening_rate > 0.0 else "hold"' in source
    assert 'history["mechanical_state_at_fraction"]' in source
    assert 'history["state_coupled_mechanical_state_at_fraction"]' in source
    assert "opening + opening_rate*float(horizon)*float(fraction)" in source
    assert "0 if opening_rate > 0.0 else 1" in source


def test_accumulated_ramp_does_not_merge_coupled_topology_fields():
    from arrhenius_fracture import global_physical_scheduler_v1 as scheduler

    source = inspect.getsource(scheduler._advance_accumulated_ramp_interval)
    assert "endpoint if callable(coupled_callback)" in source
