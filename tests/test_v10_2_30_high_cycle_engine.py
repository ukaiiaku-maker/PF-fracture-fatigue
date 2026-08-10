from types import SimpleNamespace

import numpy as np
import copy
import pytest

from arrhenius_fracture import persistent_site_high_cycle_engine_v10230 as high
from arrhenius_fracture.persistent_site_first_passage_locator_v10230 import (
    localize_first_passage,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    capture_ledgers,
    capture_stochastic_state,
    restore_active_state,
    serialize_active_state,
)
from arrhenius_fracture.persistent_site_periodic_solver_v10230 import (
    solve_periodic_state,
)
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map


class MPZ:
    def __init__(self):
        shape = (1, 2)
        self.mobile_positive = np.zeros(shape)
        self.mobile_negative = np.zeros(shape)
        self.retained_positive = np.zeros(shape)
        self.retained_negative = np.zeros(shape)
        self.accumulated_slip_positive = np.zeros(shape)
        self.accumulated_slip_negative = np.zeros(shape)
        self.wake_mobile_positive = np.zeros((1, 1))
        self.wake_mobile_negative = np.zeros((1, 1))
        self.wake_retained_positive = np.zeros((1, 1))
        self.wake_retained_negative = np.zeros((1, 1))
        self.wake_slip_positive = np.zeros((1, 1))
        self.wake_slip_negative = np.zeros((1, 1))
        self.mobile = np.zeros(shape)
        self.retained = np.zeros(shape)
        self.accumulated_slip = np.zeros(shape)
        self.wake_mobile = np.zeros((1, 1))
        self.wake_retained = np.zeros((1, 1))
        self.wake_slip = np.zeros((1, 1))
        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.signed_source_activations_total = 0.0
        self.signed_line_content_emitted_total = 0.0
        self.continuum_source_last_sigma_back_Pa = 0.0
        self.continuum_source_last_aggregate_hazard_s = 0.0
        self.persistent_site_last_geometry = {
            "front_width_m": 1.0e-6,
            "multiplicity_per_system": 1.0,
        }
        self.advance_total_m = 0.0
        self.n_bins = 2
        self.wake_n_bins = 1

    @property
    def mobile_count(self):
        return float(np.sum(self.mobile_positive + self.mobile_negative))

    @property
    def retained_count(self):
        return float(np.sum(self.retained_positive + self.retained_negative))


class Engine:
    def __init__(self, *, lambda_per_s=1.0e-15, threshold=1.0, drift_per_cycle=0.0):
        self.mpz = MPZ()
        self.lambda_per_s = float(lambda_per_s)
        self.drift_per_cycle = float(drift_per_cycle)
        self.B = 0.0
        self.hazard_threshold_action = float(threshold)
        self.hazard_action_current = 0.0
        self.hazard_event_index = 0
        self.hazard_threshold_history = []
        self.hazard_last_progress_rate_s = 0.0
        self._hazard_rng = np.random.default_rng(1720)
        self._energy_gate_pending = None
        self._energy_gate_provisional = False
        self._engine_id = 91
        self._anisotropic_tau_signed_Pa = np.asarray([1.0])
        self.t = 0.0
        self.W_emit = 0.0
        self.K_prev = 1.0
        self.N_em = 0.0
        self.n_adv = 0
        self.a_adv = 0.0
        self.micro_advance_total_m = 0.0
        self.checkpoint_advance_total_m = 0.0

    def r_eff(self):
        return 1.0e-6

    def K_shield(self):
        return 0.0

    def sigma_tip(self, K):
        return 1.0e9

    def lambda_cleave(self, sigma, temperature):
        return self.lambda_per_s, self.lambda_per_s, 1.0

    def _sync(self):
        self.mpz.mobile = self.mpz.mobile_positive + self.mpz.mobile_negative
        self.mpz.retained = self.mpz.retained_positive + self.mpz.retained_negative
        self.mpz.accumulated_slip = (
            self.mpz.accumulated_slip_positive
            + self.mpz.accumulated_slip_negative
        )
        self.mpz.continuum_source_last_sigma_back_Pa = (
            self.mpz.mobile_count * 1.0e6
        )
        self.mpz.continuum_source_last_aggregate_hazard_s = max(
            self.drift_per_cycle, 0.0
        )

    def _plastic_half_step(self, dt, temperature, sigma):
        amount = self.drift_per_cycle * float(dt)
        self.mpz.mobile_positive[0, 0] += amount
        self.mpz.accumulated_slip_positive[0, 0] += amount
        self.mpz.emitted_total += amount
        self.mpz.signed_source_activations_total += amount
        self.mpz.signed_line_content_emitted_total += amount
        self.N_em += amount
        self._sync()
        return {"dN_emit": amount, "dN_escaped": 0.0}

    def _integrate_coupled(
        self, K, T, dt, stress_override=None, lambda_override=None
    ):
        requested = max(float(dt), 0.0)
        lam = max(
            float(self.lambda_per_s if lambda_override is None else lambda_override),
            0.0,
        )
        threshold = max(float(self.hazard_threshold_action), 1.0e-300)
        remaining_action = max(threshold - self.hazard_action_current, 0.0)
        consumed = requested if lam <= 0.0 else min(requested, remaining_action / lam)
        first = self._plastic_half_step(0.5 * consumed, T, stress_override or 1.0e9)
        second = self._plastic_half_step(0.5 * consumed, T, stress_override or 1.0e9)
        action = lam * consumed
        dB = action / threshold
        self.hazard_action_current += action
        self.B = min(self.hazard_action_current / threshold, 1.0)
        self.t += consumed
        fired = self.hazard_action_current >= threshold - 1.0e-12
        if fired:
            self.hazard_threshold_history.append(threshold)
            self.hazard_event_index += 1
            self.hazard_action_current = 0.0
            self.B = 0.0
            self._energy_gate_pending = {"descriptor": {}}
        return {
            "fired": fired,
            "n_fire": int(fired),
            "v_crack": 0.0,
            "dB": dB,
            "physical_hazard_action_step": action,
            "da": 0.0,
            "dt_consumed": consumed,
            "dt_unused": max(requested - consumed, 0.0),
            "packet_mean": 0.0,
            "packet_variance_m2": 0.0,
            "lambda_c": lam,
            "lambda_c_raw": lam,
            "sigma_tip": float(stress_override or 1.0e9),
            "plastic": {
                "dN_emit": first["dN_emit"] + second["dN_emit"],
                "dN_escaped": 0.0,
            },
            "advance": {},
            "microsteps": 1,
        }


@pytest.mark.parametrize("threshold", [0.35940039563036524, 0.4332087756327596])
def test_near_threshold_cycle_locator_matches_exact_and_is_transactional(threshold):
    """DBTT/Peak B=1-1e-10 states must not enter microscopic marching."""
    remaining = threshold * 1.0e-10
    engine = Engine(lambda_per_s=0.25, threshold=threshold)
    engine.hazard_action_current = threshold - remaining
    engine.B = engine.hazard_action_current / threshold
    rng_before = copy.deepcopy(engine._hazard_rng.bit_generator.state)
    geometry_before = (engine.n_adv, engine.a_adv)

    reference = copy.deepcopy(engine)
    exact = reference._integrate_coupled(1.0, 300.0, 1.0)
    reference_cycles = exact["dt_consumed"] * Waveform().frequency_Hz

    result = localize_first_passage(
        engine, Controller(), Waveform(), 300.0, 2.0
    )
    assert result["fired"] is True
    assert result["coupled_hazard_first_passage_locator"] is True
    assert result["coupled_hazard_cycles_consumed"] == pytest.approx(
        reference_cycles, rel=1.0e-12, abs=1.0e-15
    )
    assert result["coupled_hazard_locator_bracket_high"] - result[
        "coupled_hazard_locator_bracket_low"
    ] <= 1.0
    assert result["coupled_hazard_locator_trial_evaluations"] < 100
    assert engine.hazard_event_index == 1
    assert engine.hazard_threshold_history == [threshold]
    assert engine._hazard_rng.bit_generator.state == rng_before
    assert (engine.n_adv, engine.a_adv) == geometry_before


def test_locator_allows_only_provisional_event_counter_before_energy_gate():
    class ProvisionalEngine(Engine):
        def _integrate_coupled(self, *args, **kwargs):
            result = super()._integrate_coupled(*args, **kwargs)
            if result["fired"]:
                self.n_adv += 1
            return result

    engine = ProvisionalEngine(lambda_per_s=0.25, threshold=0.5)
    engine.hazard_action_current = 0.5 * (1.0 - 1.0e-10)
    engine.B = engine.hazard_action_current / 0.5
    result = localize_first_passage(
        engine, Controller(), Waveform(), 300.0, 2.0
    )
    assert result["fired"] is True
    assert engine.n_adv == 1
    assert engine.a_adv == 0.0
    assert engine.micro_advance_total_m == 0.0
    assert engine.checkpoint_advance_total_m == 0.0
    assert engine.mpz.advance_total_m == 0.0


def test_locator_commits_bounded_exact_progress_when_horizon_has_no_bracket():
    engine = Engine(lambda_per_s=0.25, threshold=1.0)
    result = localize_first_passage(
        engine, Controller(), Waveform(), 300.0, 0.5
    )
    assert result["fired"] is False
    assert result["coupled_hazard_cycles_consumed"] == pytest.approx(0.5)
    assert result["coupled_hazard_locator_failure_reason"] == "no_bracket_within_horizon"
    assert engine.hazard_action_current == pytest.approx(0.125)




class Controller:
    def __init__(self):
        self.cfg = SimpleNamespace(
            min_block_cycles=1.0e-6,
            block_cycles=1.0e12,
            max_block_cycles=1.0e12,
            cycle_block_mode="hazard_limited",
        )

    def _phases(self):
        return np.linspace(0.0, 1.0, 8, endpoint=False)


class Waveform:
    frequency_Hz = 1.0
    period_s = 1.0
    Kmax = 1.0
    R = 0.1
    DeltaK = 0.9

    def K_phase(self, phases):
        return np.ones_like(phases)


def _fast_high_cycle_env(monkeypatch):
    monkeypatch.setenv("V10230_PERIODIC_MAX_ITERATIONS", "6")
    monkeypatch.setenv("V10230_PERIODIC_RELATIVE_TOL", "1e-12")
    monkeypatch.setenv("V10230_PERIODIC_DIAGNOSTIC_TOL", "1e-12")
    monkeypatch.setenv("V10230_HIGH_CYCLE_STATIONARY_REL_TOL", "1e-12")
    monkeypatch.setenv("V10230_HIGH_CYCLE_STATIONARY_DIAGNOSTIC_TOL", "1e-12")
    monkeypatch.setenv("V10230_HIGH_CYCLE_STATIONARY_ADMISSION_DISTANCE", "1e-12")
    monkeypatch.setenv("V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS", "24")
    monkeypatch.setenv("V10230_HIGH_CYCLE_HEARTBEAT_OPERATIONS", "1000")
    monkeypatch.setenv("V10230_FORWARD_INITIAL_CYCLES", "0.01")
    monkeypatch.setenv("V10230_FORWARD_MAX_ACCEPTED_SEGMENTS", "256")
    monkeypatch.setenv("V10230_FORWARD_MAX_TRIAL_INTEGRATIONS", "2048")
    monkeypatch.setenv("V10230_FORWARD_HEARTBEAT_SEGMENTS", "10000")


def test_complete_state_round_trip_does_not_touch_ledgers_or_rng():
    engine = Engine(drift_per_cycle=1.0)
    snapshot = serialize_active_state(engine, waveform=Waveform(), temperature_K=300.0)
    ledgers = capture_ledgers(engine)
    stochastic = capture_stochastic_state(engine)
    changed = snapshot.vector + 0.25
    restore_active_state(engine, snapshot, changed)
    assert np.allclose(serialize_active_state(engine).vector, changed)
    assert capture_ledgers(engine) == ledgers
    assert capture_stochastic_state(engine) == stochastic


def test_one_cycle_map_preserves_first_passage_and_rng():
    engine = Engine(lambda_per_s=0.25, threshold=3.5, drift_per_cycle=0.5)
    stochastic = capture_stochastic_state(engine)
    result = one_cycle_map(engine, Controller(), Waveform(), 300.0)
    assert result.hazard_action_per_cycle == 0.25
    assert result.state_end.vector.shape == result.state_start.vector.shape
    assert result.stochastic_state_preserved is True
    assert capture_stochastic_state(engine) == stochastic
    assert engine.mpz.mobile_count == 0.0


def test_periodic_solver_converges_without_consuming_production_state(monkeypatch):
    _fast_high_cycle_env(monkeypatch)
    engine = Engine(lambda_per_s=1.0e-15, drift_per_cycle=0.0)
    stochastic = capture_stochastic_state(engine)
    result = solve_periodic_state(engine, Controller(), Waveform(), 300.0)
    assert result.converged is True
    assert result.residual.converged is True
    assert capture_stochastic_state(engine) == stochastic
    assert engine.t == 0.0


def test_direct_1e12_cycle_right_censor(monkeypatch):
    _fast_high_cycle_env(monkeypatch)
    engine = Engine(lambda_per_s=1.0e-15, threshold=1.0, drift_per_cycle=0.0)
    result = high.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 1.0e12
    )
    assert result["fired"] is False
    assert result["coupled_hazard_cycles_consumed"] == 1.0e12
    assert result["coupled_hazard_partial_return"] is False
    assert abs(engine.B - 1.0e-3) < 1.0e-15
    assert engine.t == 1.0e12
    assert any(row["mode"] == "stationary_tail" for row in result["coupled_hazard_modes"])


def test_first_passage_and_post_event_restart_inside_1e12_request(monkeypatch):
    _fast_high_cycle_env(monkeypatch)
    engine = Engine(lambda_per_s=0.25, threshold=3.5, drift_per_cycle=0.0)
    first = high.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 1.0e12
    )
    assert first["fired"] is True
    assert abs(first["coupled_hazard_cycles_consumed"] - 14.0) < 2.0e-5
    assert engine._energy_gate_pending is not None
    assert any(
        row["mode"] == "first_passage_cycle_locator"
        and row.get("entry_reason") == "stationary_tail_event_guard"
        for row in first["coupled_hazard_modes"]
    )
    assert not any(
        row["mode"] == "event_guard_transient"
        for row in first["coupled_hazard_modes"]
    )

    engine._energy_gate_pending = None
    engine.n_adv += 1
    engine.a_adv += 5.0e-6
    engine.micro_advance_total_m += 5.0e-6
    engine.checkpoint_advance_total_m += 5.0e-6
    engine.lambda_per_s = 1.0e-15
    engine.hazard_threshold_action = 1.0
    engine.hazard_action_current = 0.0
    engine.B = 0.0

    second = high.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 1.0e12
    )
    assert second["fired"] is False
    assert second["coupled_hazard_cycles_consumed"] == 1.0e12
    assert engine._v10230_high_cycle_cache["geometry_signature"][0] == 1


def test_linear_slow_manifold_reaches_1e12_without_false_stationarity(monkeypatch):
    _fast_high_cycle_env(monkeypatch)
    monkeypatch.setenv("V10230_PERIODIC_RELATIVE_TOL", "1e-15")
    monkeypatch.setenv("V10230_HIGH_CYCLE_STATIONARY_REL_TOL", "1e-15")
    monkeypatch.setenv("V10230_PROJECTIVE_BURST_CYCLES", "4")
    monkeypatch.setenv("V10230_PROJECTIVE_MIN_CYCLES", "4")
    monkeypatch.setenv("V10230_PROJECTIVE_INITIAL_FACTOR", "4")
    monkeypatch.setenv("V10230_PROJECTIVE_GROWTH_FACTOR", "1000")
    monkeypatch.setenv("V10230_PROJECTIVE_MAX_CYCLES", "1e12")
    monkeypatch.setenv("V10230_PROJECTIVE_DRIFT_REL_TOL", "1e-12")
    monkeypatch.setenv("V10230_PROJECTIVE_HAZARD_REL_TOL", "1e-12")
    monkeypatch.setenv("V10230_PROJECTIVE_CURVATURE_REL_TOL", "1e-12")

    engine = Engine(lambda_per_s=1.0e-30, threshold=1.0, drift_per_cycle=1.0)
    result = high.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 1.0e12
    )
    assert result["fired"] is False
    assert result["coupled_hazard_cycles_consumed"] == 1.0e12
    assert any(row["mode"] == "slow_projective" for row in result["coupled_hazard_modes"])
    assert not any(row["mode"] == "stationary_tail" for row in result["coupled_hazard_modes"])
    assert abs(engine.mpz.mobile_count - 1.0e12) / 1.0e12 < 1.0e-10
