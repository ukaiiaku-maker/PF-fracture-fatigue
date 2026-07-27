import math
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.persistent_site_coupled_hazard_v10229 import (
    integrate_state_coupled_waveform,
)


class Waveform:
    frequency_Hz = 1.0
    period_s = 1.0
    Kmax = 1.0
    R = 0.1
    DeltaK = 0.9

    def K_phase(self, phases):
        return np.ones_like(phases, dtype=float)


class Controller:
    def __init__(self):
        self.cfg = SimpleNamespace(
            min_block_cycles=1.0e-6,
            target_dB=0.1,
            target_dN_emit=10.0,
            target_dN_mobile=10.0,
            target_dN_store=10.0,
            target_dN_escape=10.0,
        )

    def _phases(self):
        return np.linspace(0.0, 1.0, 8, endpoint=False)


class MPZ:
    def __init__(self, state=0.0):
        self.emitted_total = float(state)
        self.mobile_count = float(state)
        self.retained_count = 0.0
        self.escaped_total = 0.0


class EvolvingEngine:
    def __init__(self, state=0.0, threshold=1000.0, evolve=True):
        self.state = float(state)
        self.B = 0.0
        self.hazard_threshold_action = float(threshold)
        self.mpz = MPZ(state)
        self.t = 0.0
        self.evolve = bool(evolve)

    def sigma_tip(self, K):
        return float(K) / (1.0 + 0.5 * self.state)

    def lambda_cleave(self, sigma, temperature):
        value = 0.01 * (1.0 + 9.0 * self.state)
        return value, value, 1.0

    def K_shield(self):
        return self.state

    def r_eff(self):
        return 1.0 + self.state

    def _integrate_coupled(
        self, K, T, dt, stress_override=None, lambda_override=None
    ):
        rate = max(float(lambda_override or 0.0), 0.0)
        progress = rate / self.hazard_threshold_action
        remaining = max(1.0 - self.B, 0.0)
        consumed = float(dt) if progress <= 0.0 else min(float(dt), remaining / progress)
        initial = self.state
        if self.evolve:
            self.state = 1.0 - (1.0 - self.state) * math.exp(-consumed)
        increment = self.state - initial
        self.mpz.emitted_total += increment
        self.mpz.mobile_count += increment
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
            "plastic": {"dN_emit": increment},
            "advance": {},
            "microsteps": 1,
        }


def relaxed_tolerances(monkeypatch):
    monkeypatch.setenv("V10229_COUPLED_HAZARD_LOG_TOL_DECADES", "0.10")
    monkeypatch.setenv("V10229_COUPLED_HAZARD_SIGMA_REL_TOL", "0.05")
    monkeypatch.setenv("V10229_COUPLED_HAZARD_STATE_TARGET_FRACTION", "0.25")


def test_cleavage_action_responds_to_evolving_plastic_state(monkeypatch):
    relaxed_tolerances(monkeypatch)
    engine = EvolvingEngine(threshold=1000.0)
    result = integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 10.0
    )
    frozen_preblock = 0.01 * 10.0 / 1000.0
    assert result["dB"] > 5.0 * frozen_preblock
    assert result["coupled_hazard_lambda_end_s"] > result["coupled_hazard_lambda_start_s"]
    assert result["coupled_hazard_rejected_splits"] > 0
    assert result["coupled_hazard_frozen_within_outer_block"] is False


def test_stationary_vhcf_tail_is_one_large_segment(monkeypatch):
    relaxed_tolerances(monkeypatch)
    engine = EvolvingEngine(state=1.0, threshold=1.0e30, evolve=False)
    result = integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 1.0e12
    )
    assert result["fired"] is False
    assert result["dt_consumed"] == 1.0e12
    assert result["coupled_hazard_accepted_segments"] == 1
    assert result["coupled_hazard_stationary_tail_cycles"] == 1.0e12


def test_event_cycle_is_localized_on_coupled_trajectory(monkeypatch):
    relaxed_tolerances(monkeypatch)
    engine = EvolvingEngine(threshold=0.1)
    result = integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 100.0
    )
    assert result["fired"] is True
    assert 0.0 < result["dt_consumed"] < 100.0
    assert result["coupled_hazard_event_localized"] is True
    assert result["n_fire"] == 1
