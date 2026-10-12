import copy

from arrhenius_fracture.current_coupled_execution_r4 import (
    CurrentPersistentFractureAdapter,
)
from arrhenius_fracture.state_correspondence_r5 import (
    qualify_no_event_partition,
)
from arrhenius_fracture.stochastic_avalanche_tip import (
    _PENDING_GEOMETRY_EVENTS, clear_pending_geometry_events,
    pop_pending_geometry_event,
)


def test_rejected_preview_preserves_callers_pending_geometry_queue():
    clear_pending_geometry_events()
    sentinel = {"event": "callers-existing-descriptor"}
    _PENDING_GEOMETRY_EVENTS.append(copy.deepcopy(sentinel))
    adapter = CurrentPersistentFractureAdapter()
    adapter.preview(
        applied_K_Pa_sqrt_m=16.9223230932e6,
        temperature_K=900.0, duration_s=0.01,
    )
    assert pop_pending_geometry_event() == sentinel
    assert pop_pending_geometry_event() is None


def test_no_event_outer_source_coupling_qualification(tmp_path):
    result = qualify_no_event_partition(
        tmp_path / "partition.json", partitions=(1, 2, 4, 8)
    )
    assert result["passed"]
    assert result["qualification"]["all_event_free"]
    assert result["diagnosis"]["one_call_N_em"] > result["diagnosis"]["two_call_N_em"]
