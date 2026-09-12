from types import SimpleNamespace
import math
import pytest
from arrhenius_fracture.source_quality_transaction_v1 import source_transfer_budget
from arrhenius_fracture.source_resolution_protocol_v1 import ALLOCATED_LOG_RATE_BUDGET
from arrhenius_fracture.sharp_front import KB


def state():
    return SimpleNamespace(competition=('candidate', 'threshold'), rng_state='same_rng',
        void_state=SimpleNamespace(cavities=[SimpleNamespace(cavity_id='v', connection_exit_m=(1., 0.), connection_direction_xy=(1., 0.))]))


@pytest.mark.parametrize('error,expected', [(0., True), (.01, True), (.03, False)])
def test_prospective_log_rate_barrier_and_time_budget(monkeypatch, error, expected):
    def rate(state, tensor, temperature_K):
        lam = math.exp(tensor)
        return [{'candidate_id': 'c', 'effective_rate_s': lam,
                 'hazard_barrier_J': (1.-tensor)*KB*temperature_K, 'crossing_time_s': 1./lam}]
    monkeypatch.setattr('arrhenius_fracture.voiding_production_v5.directional_clock_rates', rate)
    result = source_transfer_budget(state(), state(), error, 0.)
    assert result['passed'] is expected
    assert result['allocated_log_rate_limit'] == ALLOCATED_LOG_RATE_BUDGET


def test_zero_reference_is_not_positive_qualification(monkeypatch):
    monkeypatch.setattr('arrhenius_fracture.voiding_production_v5.directional_clock_rates',
        lambda *a, **k: [{'candidate_id': 'c', 'effective_rate_s': 0.}])
    result = source_transfer_budget(state(), state(), None, None)
    assert result['passed'] and not result['positive_candidate_exists']
    assert 'log_rate_error' not in result['rows'][0]


def test_owned_threshold_rng_change_fails_before_rates():
    first, second = state(), state(); second.rng_state = 'redrawn'
    assert not source_transfer_budget(first, second, None, None)['passed']
