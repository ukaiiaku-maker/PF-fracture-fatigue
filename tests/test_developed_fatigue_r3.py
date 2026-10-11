import numpy as np

from arrhenius_fracture.developed_fatigue_r3 import (
    advance_mean_cycle_complete, build_developed_engine,
)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.stochastic_avalanche_tip import (
    clear_pending_geometry_events,
)


def test_mean_cycle_consumes_remainder_and_does_not_double_translate():
    clear_pending_geometry_events()
    engine = build_developed_engine()
    waveform = FatigueWaveform(Kmax=20.0e6, R=0.1, frequency_Hz=10.0)
    events = []
    for _ in range(14):
        result = advance_mean_cycle_complete(
            engine, waveform, temperature_K=1800.0, phase_count=8,
        )
        assert result["unused_cycle_time_s"] == 0.0
        events.extend(result["events"])
        if events:
            break
    assert len(events) == 1
    event = events[0]
    assert event["remaining_waveform_continued"] is True
    assert event["outer_geometry_translation_m"] == 0.0
    assert abs(event["cumulative_mpz_checkpoint_error_m"]) <= 2.0e-14
    assert event["mpz_translation_in_final_subcall_m"] <= event["event_advance_m"]
