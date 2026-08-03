from types import SimpleNamespace

import numpy as np


class MPZ:
    def __init__(self):
        shape = (1, 1)
        for name in (
            "mobile_positive",
            "mobile_negative",
            "retained_positive",
            "retained_negative",
            "accumulated_slip_positive",
            "accumulated_slip_negative",
            "wake_mobile_positive",
            "wake_mobile_negative",
            "wake_retained_positive",
            "wake_retained_negative",
            "wake_slip_positive",
            "wake_slip_negative",
        ):
            setattr(self, name, np.zeros(shape, dtype=float))
        self.mobile = np.zeros(shape)
        self.retained = np.zeros(shape)
        self.accumulated_slip = np.zeros(shape)
        self.wake_mobile = np.zeros(shape)
        self.wake_retained = np.zeros(shape)
        self.wake_slip = np.zeros(shape)
        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.signed_source_activations_total = 0.0
        self.signed_line_content_emitted_total = 0.0
        self.continuum_source_last_sigma_back_Pa = 0.0
        self.continuum_source_last_aggregate_hazard_s = 1.0
        self.persistent_site_last_geometry = {
            "front_width_m": 1.0e-6,
            "multiplicity_per_system": 1.0,
        }
        self.advance_total_m = 0.0
        self.n_bins = 1
        self.wake_n_bins = 1

    @property
    def mobile_count(self):
        return float(np.sum(self.mobile_positive + self.mobile_negative))

    @property
    def retained_count(self):
        return float(np.sum(self.retained_positive + self.retained_negative))


class AffineEngine:
    def __init__(self, drift=1.0, relaxation=0.0, target=0.0, hazard=1.0e-30):
        self.mpz = MPZ()
        self.drift = float(drift)
        self.relaxation = float(relaxation)
        self.target = float(target)
        self.hazard = float(hazard)
        self.B = 0.0
        self.hazard_threshold_action = 1.0
        self.hazard_action_current = 0.0
        self.hazard_event_index = 0
        self.hazard_threshold_history = []
        self.hazard_last_progress_rate_s = 0.0
        self._hazard_rng = np.random.default_rng(1720)
        self._energy_gate_pending = None
        self._energy_gate_provisional = False
        self._anisotropic_tau_signed_Pa = np.asarray([1.0])
        self._engine_id = 7
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
        rate = self.hazard * (1.0 + 1.0e-6 * self.mpz.mobile_count)
        return rate, rate, 1.0

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

    def _plastic_half_step(self, dt, temperature, sigma):
        dt = float(dt)
        old = float(self.mpz.mobile_positive[0, 0])
        if self.relaxation > 0.0:
            decay = np.exp(-self.relaxation * dt)
            new = self.target + (old - self.target) * decay + self.drift * dt
        else:
            new = old + self.drift * dt
        increment = new - old
        self.mpz.mobile_positive[0, 0] = new
        self.mpz.accumulated_slip_positive[0, 0] += max(increment, 0.0)
        self.mpz.emitted_total += max(increment, 0.0)
        self.mpz.signed_source_activations_total += increment
        self.mpz.signed_line_content_emitted_total += increment
        self.N_em += max(increment, 0.0)
        self._sync()
        return {"dN_emit": max(increment, 0.0), "dN_escaped": 0.0}


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


def configure(monkeypatch):
    monkeypatch.setenv("V10230_DMD_BURST_CYCLES", "12")
    monkeypatch.setenv("V10230_DMD_MAX_RANK", "6")
    monkeypatch.setenv("V10230_DMD_MIN_PROJECT_CYCLES", "8")
    monkeypatch.setenv("V10230_DMD_NEUTRAL_EIGEN_TOL", "1e-6")
    monkeypatch.setenv("V10230_DMD_TRAINING_REL_TOL", "1e-10")
    monkeypatch.setenv("V10230_DMD_STATE_VALIDATION_REL_TOL", "1e-9")
    monkeypatch.setenv("V10230_DMD_HAZARD_VALIDATION_REL_TOL", "1e-9")
    monkeypatch.setenv("V10230_DMD_LEDGER_VALIDATION_REL_TOL", "1e-9")
    monkeypatch.setenv("V10230_DMD_CHAIN_MAX_SEGMENTS", "96")
    monkeypatch.setenv("V10230_DMD_CHAIN_GROWTH_FACTOR", "2")
    monkeypatch.setenv("V10230_DMD_CHAIN_MIN_GROWTH_FACTOR", "1.25")
    monkeypatch.setenv("V10230_PERIODIC_RELATIVE_TOL", "1e-15")
    monkeypatch.setenv("V10230_HIGH_CYCLE_STATIONARY_REL_TOL", "1e-15")
    monkeypatch.setenv("V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS", "16")
    monkeypatch.setenv("V10230_HIGH_CYCLE_HEARTBEAT_OPERATIONS", "1000")
    monkeypatch.setenv("V10230_PROJECTIVE_BURST_CYCLES", "4")
    monkeypatch.setenv("V10230_PROJECTIVE_MIN_CYCLES", "4")
    monkeypatch.setenv("V10230_PROJECTIVE_INITIAL_FACTOR", "4")
    monkeypatch.setenv("V10230_PROJECTIVE_GROWTH_FACTOR", "2")
    monkeypatch.setenv("V10230_PROJECTIVE_MAX_CYCLES", "1e12")
