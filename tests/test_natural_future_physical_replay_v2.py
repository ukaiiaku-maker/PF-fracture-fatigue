from dataclasses import replace
import numpy as np

from arrhenius_fracture.natural_future_physical_replay_v2 import compare_states, exact_projection
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


CROSSING = {
    "real_crossing_executed": True,
    "selected_event_identity_exact": True,
    "accepted_topology_exact": True,
    "categorical_terminal_exact": True,
    "minimum_event_selection_margin_action": 0.25,
    "maximum_event_selection_perturbation_action": 0.0,
}


def test_identical_physical_states_pass_but_result_is_not_called_bitwise():
    state, _ = build_production_void_state(stochastic=True, seed=12000)
    result = compare_states(state, state, case_identity="12000", seed=12000,
                            solver_condition_number=100.0, free_residual_relative=1e-12,
                            subsequent_crossing=CROSSING)
    assert result["passed"]
    assert result["comparison_kind"] == "FUTURE_PHYSICAL_REPLAY_NOT_BITWISE_EQUALITY"


def test_one_ulp_continuous_difference_can_pass_under_recorded_numerical_budget():
    state, _ = build_production_void_state(stochastic=True, seed=12000)
    changed = np.asarray(state.displacement).copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    peer = replace(state, displacement=changed)
    result = compare_states(state, peer, case_identity="12000", seed=12000,
                            solver_condition_number=1e6, free_residual_relative=1e-12,
                            subsequent_crossing=CROSSING)
    assert result["passed"]


def test_rng_or_threshold_identity_difference_fails_exact_projection():
    state, _ = build_production_void_state(stochastic=True, seed=12000)
    peer = replace(state, rng_state={"different": True})
    assert exact_projection(state) != exact_projection(peer)
    result = compare_states(state, peer, case_identity="12000", seed=12000,
                            solver_condition_number=100.0, free_residual_relative=1e-12,
                            subsequent_crossing=CROSSING)
    assert not result["passed"]


def test_missing_subsequent_real_crossing_fails_closed():
    state, _ = build_production_void_state(stochastic=True, seed=12000)
    result = compare_states(state, state, case_identity="12000", seed=12000,
                            solver_condition_number=100.0, free_residual_relative=1e-12,
                            subsequent_crossing=None)
    assert not result["passed"]


def test_exact_projection_covers_a_real_cavity_checkpoint():
    from arrhenius_fracture.voiding_production_v5 import deterministic_trajectory
    trace = []
    deterministic_trajectory(stop_before_ligament=True, state_trace=trace)
    projected = exact_projection(dict(trace)["subgrid_void"])
    assert projected["cavity_topology"][0]["parent_site_id"] == "site-1"


def test_missing_event_selection_margin_fails_closed():
    state, _ = build_production_void_state(stochastic=True, seed=12000)
    crossing = {key: value for key, value in CROSSING.items() if "margin" not in key}
    result = compare_states(state, state, case_identity="12000", seed=12000,
                            solver_condition_number=100.0, free_residual_relative=1e-12,
                            subsequent_crossing=crossing)
    assert not result["passed"]
