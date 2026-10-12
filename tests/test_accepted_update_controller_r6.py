import copy

from arrhenius_fracture.accepted_update_controller_r6 import (
    advance_current_source_adaptive, advance_current_source_to_earliest_event,
)
from arrhenius_fracture.current_coupled_execution_r4 import (
    build_current_r4_engine, complete_state,
)
from arrhenius_fracture.state_correspondence_r5 import advance_fixed_horizon


def test_adaptive_source_refresh_is_private_and_matches_fine_hold():
    parent = build_current_r4_engine()
    before = complete_state(parent)
    accepted, receipt = advance_current_source_adaptive(
        parent, applied_K_Pa_sqrt_m=16.9223230932e6,
        temperature_K=900.0, duration_s=8.4,
    )
    assert complete_state(parent) == before
    assert receipt["accepted_subsegments"] != 128
    assert receipt["trial_integrations"] > receipt["accepted_subsegments"]

    reference = copy.deepcopy(parent)
    advance_fixed_horizon(
        reference, applied_K_Pa_sqrt_m=16.9223230932e6,
        temperature_K=900.0, duration_s=8.4,
        maximum_outer_substep_s=8.4 / 128.0,
    )
    actual_population = accepted.mpz.mobile_count + accepted.mpz.retained_count
    reference_population = reference.mpz.mobile_count + reference.mpz.retained_count
    assert abs(actual_population-reference_population) / reference_population < 0.005


def test_adaptive_event_interval_carries_later_time():
    parent = build_current_r4_engine()
    event_state, first = advance_current_source_adaptive(
        parent, applied_K_Pa_sqrt_m=51.10046751415974e6,
        temperature_K=900.0, duration_s=1.0e-6,
    )
    later_state, later = advance_current_source_adaptive(
        event_state, applied_K_Pa_sqrt_m=5.0e6,
        temperature_K=900.0, duration_s=0.05,
    )
    assert event_state.n_adv >= 1
    assert later_state.t > event_state.t
    assert first["input_owner_mutated"] is False
    assert later["input_owner_mutated"] is False


def test_event_aware_accepted_update_stops_at_native_event_and_is_private():
    parent = build_current_r4_engine()
    before = complete_state(parent)
    accepted = advance_current_source_to_earliest_event(
        parent, applied_K_Pa_sqrt_m=51.10046751415974e6,
        temperature_K=900.0, duration_s=1.0e-6,
    )
    assert complete_state(parent) == before
    assert accepted.result["fired"]
    assert accepted.consumed_s <= 1.0e-6
    assert accepted.audit["remainder_owner"] == "CALLING_COUPLED_SCHEDULER"
    assert accepted.audit["accepted_path"] == "TWO_HALF_EARLIEST_EVENT"
    assert accepted.audit["absolute_floors"]["population_and_slip"] == 100.0
    assert accepted.engine.n_adv == 1
