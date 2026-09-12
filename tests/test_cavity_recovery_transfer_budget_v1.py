import importlib.util
from pathlib import Path
import math
import pytest

spec = importlib.util.spec_from_file_location("recovery_transfer", Path(__file__).resolve().parents[1]/
    "scripts/qualify_cavity_recovery_transfer_budget_v1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rate(value, *, barrier=1e-18, candidate="owned"):
    return dict(candidate_id=candidate, effective_rate_s=value, hazard_barrier_J=barrier,
                crossing_time_s=1/value if value else math.inf)


def test_identical_positive_preserved_clock_passes():
    result = module.candidate_budget(rate(1e-100), rate(1e-100))
    assert result["passed"] and result["positive"]
    assert result["log_rate_error"] == result["waiting_time_relative_error"] == 0


def test_zero_rates_do_not_become_positive():
    result = module.candidate_budget(rate(0), rate(0))
    assert result["passed"] and not result["positive"]
    assert result["log_rate_error"] is None


@pytest.mark.parametrize("first, second", [(0., 1e-100), (1e-320, 1e-320), (math.inf, 1.), (-1., 1.)])
def test_inactive_nearzero_nonfinite_fail_closed(first, second):
    assert not module.candidate_budget(rate(first), rate(second))["passed"]


def test_tensor_like_small_rate_error_cannot_bypass_timing_allocation():
    result = module.candidate_budget(rate(1.04), rate(1.))
    assert result["log_rate_error"] > module.ALLOCATED_LOG_RATE_BUDGET
    assert not result["passed"]


def test_barrier_error_uses_kbt_not_relative_barrier():
    reference = rate(1.)
    current = rate(1., barrier=reference["hazard_barrier_J"]+module.KB*module.TEMPERATURE_K*.03)
    result = module.candidate_budget(current, reference)
    assert result["effective_barrier_error_over_kBT"] == pytest.approx(.03)
    assert not result["passed"]


def test_candidate_mismatch_fails():
    assert not module.candidate_budget(rate(1., candidate="other"), rate(1.))["passed"]


def test_changed_threshold_waiting_time_fails_even_when_rates_match():
    current = rate(1.)
    current["crossing_time_s"] = 2.
    assert not module.candidate_budget(current, rate(1.))["passed"]


def test_full_and_half_allocations_remain_distinct():
    result = module.candidate_budget(rate(1.04), rate(1.))
    assert result["total_budget_passed"] and not result["passed"]


def test_exact_owned_source_site_cannot_be_approximated_away():
    base = {key: [] for key in ("candidate_records", "hazard_records", "rng_state",
        "normal_xy", "tangent_xy", "pending_event_ids", "consumed_event_ids")}
    base.update(tensor_Pa=[[1., 0.], [0., 1.]], owned_source_identity={"x": 1.},
                stored_raw_source_provenance={"point": [1., 0.]}, operator_id="same",
                tensor_fingerprint="same", unguarded_candidate_rates=[rate(1.)])
    current = dict(base, owned_source_identity={"x": 1.+1e-12})
    result = module.compare_sources(current, base)
    assert result["tensor_gate"]
    assert result["candidate_rows"][0]["passed"]
    assert not result["exact_identity_checks"]["owned_source_identity"]
    assert not result["allocated_transfer_budget_passed"]
