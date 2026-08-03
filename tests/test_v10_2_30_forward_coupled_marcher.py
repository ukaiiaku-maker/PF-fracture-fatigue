import math
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.persistent_site_forward_coupled_hazard_v10230 import (
    integrate_state_coupled_waveform,
)
from arrhenius_fracture.persistent_site_forward_selector_v10230 import (
    select_forward_block,
)


class Waveform:
    frequency_Hz = 1.0
    period_s = 1.0
    Kmax = 10.0
    R = 0.1
    DeltaK = 9.0

    def K_phase(self, phases):
        return np.full_like(phases, self.Kmax, dtype=float)


class Controller:
    def __init__(self):
        self.cfg = SimpleNamespace(
            min_block_cycles=1.0e-6,
            target_dB=0.1,
            target_dN_emit=0.1,
            target_dN_mobile=0.1,
            target_dN_store=0.1,
            target_dN_escape=0.1,
            block_cycles=1.0e10,
            max_block_cycles=1.0e10,
            cycle_block_mode="hazard_limited",
        )

    def _phases(self):
        return np.linspace(0.0, 1.0, 8, endpoint=False)


class MPZ:
    def __init__(self, state=0.0):
        self.emitted_total = float(state)
        self.mobile_count = float(state)
        self.retained_count = 0.0
        self.escaped_total = 0.0


class ConstantEngine:
    persistent_site_cyclic_v10229 = True

    def __init__(self, rate=0.01, threshold=1.0e30):
        self.rate = float(rate)
        self.B = 0.0
        self.hazard_threshold_action = float(threshold)
        self.mpz = MPZ()
        self.t = 0.0
        self._v10229_last_vhcf_block_audit = None

    def sigma_tip(self, K):
        return float(K) - self.K_shield()

    def lambda_cleave(self, sigma, temperature):
        return self.rate, self.rate, 1.0

    def K_shield(self):
        return 2.0

    def r_eff(self):
        return 1.0

    def _integrate_coupled(
        self, K, T, dt, stress_override=None, lambda_override=None
    ):
        rate = max(float(lambda_override or 0.0), 0.0)
        progress = rate / self.hazard_threshold_action
        remaining = max(1.0 - self.B, 0.0)
        consumed = float(dt) if progress <= 0.0 else min(float(dt), remaining / progress)
        dB = min(progress * consumed, remaining)
        self.B += dB
        fired = self.B >= 1.0 - 1.0e-12
        if fired:
            self.B = 0.0
        self.t += consumed
        return {
            "fired": fired,
            "n_fire": int(fired),
            "v_crack": 0.0,
            "dB": dB,
            "physical_hazard_action_step": dB * self.hazard_threshold_action,
            "da": 0.0,
            "dt_consumed": consumed,
            "dt_unused": max(float(dt) - consumed, 0.0),
            "packet_mean": 0.0,
            "packet_variance_m2": 0.0,
            "lambda_c": rate,
            "lambda_c_raw": rate,
            "Gc_J": 1.0,
            "sigma_tip": float(stress_override or 0.0),
            "plastic": {},
            "advance": {},
            "microsteps": 1,
        }


class EvolvingEngine(ConstantEngine):
    def __init__(self):
        super().__init__(rate=0.01, threshold=1.0e30)
        self.state = 0.0
        self.mpz = MPZ()

    def sigma_tip(self, K):
        return float(K) / (1.0 + self.state)

    def lambda_cleave(self, sigma, temperature):
        rate = 0.01 * (1.0 + 4.0 * self.state)
        return rate, rate, 1.0

    def K_shield(self):
        return self.state

    def r_eff(self):
        return 1.0 + self.state

    def _integrate_coupled(
        self, K, T, dt, stress_override=None, lambda_override=None
    ):
        result = super()._integrate_coupled(
            K,
            T,
            dt,
            stress_override=stress_override,
            lambda_override=lambda_override,
        )
        consumed = float(result["dt_consumed"])
        initial = self.state
        self.state = 1.0 - (1.0 - self.state) * math.exp(-consumed)
        increment = self.state - initial
        self.mpz.emitted_total += increment
        self.mpz.mobile_count += increment
        result["plastic"] = {"dN_emit": increment}
        return result


def _set_common_forward_env(monkeypatch):
    monkeypatch.setenv("V10230_FORWARD_INITIAL_CYCLES", "0.01")
    monkeypatch.setenv("V10230_FORWARD_GROWTH_FACTOR", "2")
    monkeypatch.setenv("V10230_FORWARD_MAX_ACCEPTED_SEGMENTS", "512")
    monkeypatch.setenv("V10230_FORWARD_MAX_TRIAL_INTEGRATIONS", "4096")
    monkeypatch.setenv("V10230_FORWARD_HEARTBEAT_SEGMENTS", "10000")


def test_constant_state_semigroup_is_exact(monkeypatch):
    _set_common_forward_env(monkeypatch)
    controller = Controller()
    waveform = Waveform()

    one = ConstantEngine(rate=0.01, threshold=1.0e30)
    one_result = integrate_state_coupled_waveform(
        one, controller, waveform, 300.0, 100.0
    )

    split = ConstantEngine(rate=0.01, threshold=1.0e30)
    split_dB = 0.0
    for _ in range(10):
        result = integrate_state_coupled_waveform(
            split, controller, waveform, 300.0, 10.0
        )
        split_dB += result["dB"]

    assert one_result["coupled_hazard_forward_marcher"] is True
    assert one_result["coupled_hazard_recursive_bisection"] is False
    assert one_result["coupled_hazard_two_half_step_state_committed"] is True
    assert one_result["coupled_hazard_third_commit_integration"] is False
    assert one_result["coupled_hazard_work_budget_exhausted"] is False
    assert math.isclose(one.t, split.t, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(one.B, split.B, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(one_result["dB"], split_dB, rel_tol=0.0, abs_tol=1.0e-15)


def test_event_cycle_is_block_size_independent(monkeypatch):
    _set_common_forward_env(monkeypatch)
    monkeypatch.setenv("V10230_FORWARD_EVENT_LOCALIZATION_CYCLES", "1e-5")
    controller = Controller()
    waveform = Waveform()
    expected = 3.7

    event_cycles = []
    for requested in (10.0, 100.0, 1000.0):
        engine = ConstantEngine(rate=1.0, threshold=expected)
        result = integrate_state_coupled_waveform(
            engine, controller, waveform, 300.0, requested
        )
        assert result["fired"] is True
        assert result["coupled_hazard_event_localized"] is True
        event_cycles.append(result["coupled_hazard_cycles_consumed"])

    for value in event_cycles:
        assert math.isclose(value, expected, rel_tol=0.0, abs_tol=2.0e-5)
    assert max(event_cycles) - min(event_cycles) <= 2.0e-5


def test_work_budget_returns_partial_progress(monkeypatch):
    _set_common_forward_env(monkeypatch)
    monkeypatch.setenv("V10230_FORWARD_MAX_ACCEPTED_SEGMENTS", "2")
    monkeypatch.setenv("V10230_FORWARD_MAX_TRIAL_INTEGRATIONS", "12")
    controller = Controller()
    engine = EvolvingEngine()
    result = integrate_state_coupled_waveform(
        engine, controller, Waveform(), 300.0, 1.0e6
    )

    assert result["fired"] is False
    assert result["coupled_hazard_work_budget_exhausted"] is True
    assert result["coupled_hazard_partial_return"] is True
    assert 0.0 < result["coupled_hazard_cycles_consumed"] < 1.0e6
    assert result["dt_unused"] > 0.0
    assert result["coupled_hazard_accepted_segments"] <= 2
    assert result["coupled_hazard_trial_integrations"] <= 12


def test_forward_selector_is_predictor_only(monkeypatch):
    monkeypatch.setenv("V10230_FORWARD_OUTER_PROPOSAL_CYCLES", "1e6")
    engine = ConstantEngine()
    prediction = SimpleNamespace(_v10229_vhcf_engine=engine)
    result = select_forward_block(
        Controller(), prediction, 1.0e10, {"cycles": 0.001}
    )

    assert result["cycles"] == 1.0e6
    assert result["limiter"] == "forward_marcher_proposal_cap"
    audit = engine._v10229_last_vhcf_block_audit
    assert audit["private_trial_evaluations"] == 0
    assert audit["raw_population_targets_used"] is False
    assert audit["committer_authoritative"] is True
