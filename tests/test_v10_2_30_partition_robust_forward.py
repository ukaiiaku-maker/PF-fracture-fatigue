from types import SimpleNamespace

import numpy as np

from arrhenius_fracture import persistent_site_forward_robust_v10230 as robust


class MPZ:
    def __init__(self, mobile=0.0, retained=0.0, backstress=0.0):
        self.mobile_count = float(mobile)
        self.retained_count = float(retained)
        self.emitted_total = float(mobile)
        self.escaped_total = 0.0
        self.continuum_source_last_sigma_back_Pa = float(backstress)
        self.continuum_source_last_aggregate_hazard_s = 1.0
        self.persistent_site_last_geometry = {
            "front_width_m": 1.0e-6,
            "multiplicity_per_system": 10.0,
        }
        self.mobile_positive = np.asarray([[mobile]], dtype=float)
        self.mobile_negative = np.zeros((1, 1), dtype=float)
        self.retained_positive = np.asarray([[retained]], dtype=float)
        self.retained_negative = np.zeros((1, 1), dtype=float)
        self.accumulated_slip_positive = np.asarray([[mobile]], dtype=float)
        self.accumulated_slip_negative = np.zeros((1, 1), dtype=float)


class Engine:
    persistent_site_cyclic_v10229 = True

    def __init__(self, mobile=0.0):
        self.mpz = MPZ(mobile=mobile, backstress=mobile * 1.0e6)
        self.B = 0.0
        self.t = 0.0

    def sigma_tip(self, K):
        return 10.0

    def lambda_cleave(self, sigma, temperature):
        return 1.0e-6, 1.0e-6, 1.0

    def K_shield(self):
        return 0.0

    def r_eff(self):
        return 1.0e-6

    def _integrate_coupled(self, K, T, dt, stress_override=None, lambda_override=None):
        increment = float(dt) ** 2
        self.mpz.mobile_count += increment
        self.mpz.emitted_total += increment
        self.mpz.mobile_positive[0, 0] += increment
        self.mpz.accumulated_slip_positive[0, 0] += increment
        self.mpz.continuum_source_last_sigma_back_Pa = self.mpz.mobile_count * 1.0e6
        self.B += max(float(lambda_override or 0.0), 0.0) * float(dt)
        self.t += float(dt)
        return {
            "fired": False,
            "n_fire": 0,
            "v_crack": 0.0,
            "dB": max(float(lambda_override or 0.0), 0.0) * float(dt),
            "physical_hazard_action_step": 0.0,
            "da": 0.0,
            "dt_consumed": float(dt),
            "dt_unused": 0.0,
            "packet_mean": 0.0,
            "packet_variance_m2": 0.0,
            "lambda_c": max(float(lambda_override or 0.0), 0.0),
            "lambda_c_raw": max(float(lambda_override or 0.0), 0.0),
            "sigma_tip": float(stress_override or 0.0),
            "plastic": {"dN_emit": increment},
            "advance": {},
            "microsteps": 1,
        }


class Controller:
    def __init__(self):
        self.cfg = SimpleNamespace(
            min_block_cycles=1.0e-6,
            block_cycles=1.0e6,
            max_block_cycles=1.0e6,
            cycle_block_mode="hazard_limited",
        )

    def _phases(self):
        return np.linspace(0.0, 1.0, 4, endpoint=False)


class Waveform:
    frequency_Hz = 1.0
    period_s = 1.0
    Kmax = 10.0
    R = 0.1
    DeltaK = 9.0

    def K_phase(self, phases):
        return np.full_like(phases, self.Kmax, dtype=float)


def _config():
    return {
        "state_profile_relative_tol": 1.0e-4,
        "mobile_relative_tol": 1.0e-4,
        "retained_relative_tol": 1.0e-4,
        "backstress_relative_tol": 1.0e-4,
        "emission_log_rate_tol_decades": 0.01,
    }


def test_active_state_error_detects_full_vs_half_disagreement():
    full = Engine(mobile=100.0)
    half = Engine(mobile=101.0)
    metrics = robust._state_error_metrics(full, half, _config())
    assert metrics["mobile_relative_error"] > 0.009
    assert metrics["backstress_relative_error"] > 0.009
    assert metrics["state_profile_relative_error"] > 0.009
    assert metrics["state_maximum_error_ratio"] > 90.0
    assert metrics["state_limiting_error"] in {
        "state_profile",
        "mobile",
        "backstress",
    }


def test_segment_size_is_carried_across_outer_calls(monkeypatch):
    monkeypatch.setenv("V10230_FORWARD_INITIAL_CYCLES", "0.01")
    monkeypatch.setenv("V10230_FORWARD_MAX_SEGMENT_CYCLES", "10")
    monkeypatch.setenv("V10230_FORWARD_STATE_PROFILE_REL_TOL", "0.2")
    monkeypatch.setenv("V10230_FORWARD_MOBILE_REL_TOL", "0.2")
    monkeypatch.setenv("V10230_FORWARD_BACKSTRESS_REL_TOL", "0.2")
    monkeypatch.setenv("V10230_FORWARD_RETAINED_REL_TOL", "0.2")
    monkeypatch.setenv("V10230_FORWARD_MAX_ACCEPTED_SEGMENTS", "64")
    monkeypatch.setenv("V10230_FORWARD_MAX_TRIAL_INTEGRATIONS", "512")
    monkeypatch.setenv("V10230_FORWARD_HEARTBEAT_SEGMENTS", "10000")

    engine = Engine()
    first = robust.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 0.02
    )
    carried = engine._v10230_forward_next_segment_cycles
    assert first["coupled_hazard_partition_robust_state_control"] is True
    assert carried > 0.01

    second = robust.integrate_state_coupled_waveform(
        engine, Controller(), Waveform(), 300.0, 0.02
    )
    first_segment = second["coupled_hazard_segments"][0]["cycles_proposed"]
    assert first_segment >= min(carried, 0.02) - 1.0e-15


def test_explicit_transient_segment_cap_is_enforced(monkeypatch):
    monkeypatch.setenv("V10230_FORWARD_INITIAL_CYCLES", "100")
    monkeypatch.setenv("V10230_FORWARD_MAX_SEGMENT_CYCLES", "0.05")
    monkeypatch.setenv("V10230_FORWARD_STATE_PROFILE_REL_TOL", "1")
    monkeypatch.setenv("V10230_FORWARD_MOBILE_REL_TOL", "1")
    monkeypatch.setenv("V10230_FORWARD_BACKSTRESS_REL_TOL", "1")
    monkeypatch.setenv("V10230_FORWARD_RETAINED_REL_TOL", "1")
    monkeypatch.setenv("V10230_FORWARD_MAX_ACCEPTED_SEGMENTS", "16")
    monkeypatch.setenv("V10230_FORWARD_MAX_TRIAL_INTEGRATIONS", "128")
    monkeypatch.setenv("V10230_FORWARD_HEARTBEAT_SEGMENTS", "10000")

    result = robust.integrate_state_coupled_waveform(
        Engine(), Controller(), Waveform(), 300.0, 0.1
    )
    assert result["coupled_hazard_config"]["maximum_segment_cycles"] == 0.05
    assert all(
        row["cycles_proposed"] <= 0.05 + 1.0e-15
        for row in result["coupled_hazard_segments"]
    )
