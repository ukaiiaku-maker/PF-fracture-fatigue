import copy
import math

import pytest

from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate,
    DirectionalHazardState,
    commit_directional_interval,
    directional_drive,
    preview_directional_interval,
    preview_production_cleavage_rate,
    tungsten_cleavage_candidates,
)


def candidate():
    return CleavageCandidate.create(
        plane_family="cleavage",
        plane_variant="(100)",
        direction_xy=(1.0, -0.0),
        normal_xy=(0.0, 1.0),
        gamma_rel=1.0,
        orientation_convention="bcc-[001]:theta=0",
    )


@pytest.mark.parametrize(
    "action, rate, duration, expected_count, expected_residual, expected_times",
    [
        (0.0, 0.0, 10.0, 0, 0.0, ()),
        (0.0, 2.0, 0.0, 0, 0.0, ()),
        (0.0, 0.25, 2.0, 0, 0.5, ()),
        (0.25, 1.0, 0.5, 0, 0.75, ()),
        (0.999999, 0.001, 0.002, 1, 0.000001, (5.001,)),
        (0.25, 2.0, 1.5, 3, 0.25, (5.375, 5.875, 6.375)),
        (2.25, 2.0, 0.5, 1, 0.25, (5.375,)),
    ],
)
def test_exact_constant_rate_solution(
    action, rate, duration, expected_count, expected_residual, expected_times
):
    state = DirectionalHazardState.from_action(candidate().candidate_id, action)
    preview = preview_directional_interval(
        state, lambda_per_s=rate, start_time_s=5.0, duration_s=duration
    )
    accepted = commit_directional_interval(state, preview)

    assert state.action == action  # preview and immutable commit leave input untouched
    assert accepted.action == pytest.approx(action + rate * duration)
    assert accepted.completed_event_count == state.completed_event_count + expected_count
    assert accepted.residual_action == pytest.approx(expected_residual, abs=2e-12)
    assert tuple(event.completion_time_s for event in preview.completed_events) == pytest.approx(
        expected_times
    )


def test_single_candidate_parity_with_scalar_first_passage():
    state = DirectionalHazardState.from_action(candidate().candidate_id, 0.4)
    rate, start, duration = 3.0, 7.0, 1.2
    preview = preview_directional_interval(
        state, lambda_per_s=rate, start_time_s=start, duration_s=duration
    )
    accepted = commit_directional_interval(state, preview)
    scalar_end = 0.4 + rate * duration
    scalar_count = math.floor(scalar_end + 1.0e-13) - math.floor(0.4 + 1.0e-13)

    assert accepted.action == scalar_end
    assert len(preview.completed_events) == scalar_count
    assert accepted.residual_action == pytest.approx(scalar_end - math.floor(scalar_end + 1.0e-13))
    assert [event.completion_time_s for event in preview.completed_events] == pytest.approx(
        [start + (threshold - 0.4) / rate for threshold in range(1, 5)]
    )


@pytest.mark.parametrize("signed_J", [-10.0, 0.0])
def test_nonpositive_signed_J_has_no_topology_drive(signed_J):
    result = directional_drive(candidate(), signed_J_J_per_m2=signed_J, Eprime_Pa=4e11)
    assert result.positive_J_J_per_m2 == 0.0
    assert result.K_directional_Pa_sqrt_m == 0.0
    assert result.lambda_per_s == 0.0


def test_positive_signed_J_maps_exactly_to_K():
    result = directional_drive(candidate(), signed_J_J_per_m2=25.0, Eprime_Pa=4e11)
    assert result.K_directional_Pa_sqrt_m == pytest.approx(math.sqrt(1e13))


def test_production_rate_preview_uses_engine_without_mutation():
    class Engine:
        def __init__(self):
            self.marker = {"accepted": 7}

        @staticmethod
        def sigma_tip(K):
            return 2.0 * K

        @staticmethod
        def lambda_cleave(stress, temperature):
            return stress / temperature, stress / temperature, 1.0

    engine = Engine()
    before = copy.deepcopy(engine.__dict__)
    first = preview_production_cleavage_rate(
        engine, candidate(), signed_J_J_per_m2=4.0, Eprime_Pa=9.0, temperature_K=3.0
    )
    second = preview_production_cleavage_rate(
        engine, candidate(), signed_J_J_per_m2=4.0, Eprime_Pa=9.0, temperature_K=3.0
    )
    assert first == second
    assert first.lambda_per_s == 4.0
    assert engine.__dict__ == before


def test_tungsten_adapter_uses_exact_forward_admissibility_only():
    candidates = tungsten_cleavage_candidates(theta_deg=30.0)
    assert candidates
    assert all(item.direction_xy[0] > 0.0 for item in candidates)
    assert {item.plane_variant for item in candidates}.issubset({"(100)", "(010)"})
