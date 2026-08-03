from types import SimpleNamespace

import numpy as np

from arrhenius_fracture import persistent_site_high_cycle_engine_v10230_v2 as high


class _MPZ:
    def __init__(self):
        self.mobile_positive = np.zeros((1, 1))
        self.mobile_negative = np.zeros((1, 1))
        self.retained_positive = np.zeros((1, 1))
        self.retained_negative = np.zeros((1, 1))
        self.accumulated_slip_positive = np.zeros((1, 1))
        self.accumulated_slip_negative = np.zeros((1, 1))
        self.mobile = np.zeros((1, 1))
        self.retained = np.zeros((1, 1))
        self.accumulated_slip = np.zeros((1, 1))
        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.signed_source_activations_total = 0.0
        self.signed_line_content_emitted_total = 0.0
        self.continuum_source_last_sigma_back_Pa = 0.0
        self.continuum_source_last_aggregate_hazard_s = 1.0
        self.advance_total_m = 0.0
        self.n_bins = 1
        self.wake_n_bins = 0

    @property
    def mobile_count(self):
        return float(np.sum(self.mobile_positive + self.mobile_negative))

    @property
    def retained_count(self):
        return float(np.sum(self.retained_positive + self.retained_negative))


class _Engine:
    def __init__(self):
        self.mpz = _MPZ()
        self.B = 0.0
        self.hazard_threshold_action = 1.0
        self.hazard_action_current = 0.0
        self.hazard_event_index = 0
        self.hazard_threshold_history = []
        self.hazard_last_progress_rate_s = 0.0
        self._hazard_rng = np.random.default_rng(1720)
        self._energy_gate_pending = None
        self._energy_gate_provisional = False
        self._engine_id = 77
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

    def sigma_tip(self, _K):
        return 1.0e9

    def lambda_cleave(self, _sigma, _temperature):
        return 1.0e-30, 1.0e-30, 1.0

    def _plastic_half_step(self, dt, _temperature, _sigma):
        amount = 0.5 * float(dt)
        self.mpz.mobile_positive[0, 0] += amount
        self.mpz.accumulated_slip_positive[0, 0] += amount
        self.mpz.mobile = self.mpz.mobile_positive + self.mpz.mobile_negative
        self.mpz.retained = self.mpz.retained_positive + self.mpz.retained_negative
        self.mpz.accumulated_slip = (
            self.mpz.accumulated_slip_positive
            + self.mpz.accumulated_slip_negative
        )
        self.mpz.emitted_total += amount
        self.mpz.signed_source_activations_total += amount
        self.mpz.signed_line_content_emitted_total += amount
        self.N_em += amount
        self.mpz.continuum_source_last_sigma_back_Pa = self.mpz.mobile_count
        return {"dN_emit": amount}


class _Controller:
    cfg = SimpleNamespace(min_block_cycles=1.0e-6)

    def _phases(self):
        return np.linspace(0.0, 1.0, 4, endpoint=False)


class _Waveform:
    frequency_Hz = 1.0
    period_s = 1.0
    Kmax = 1.0

    def K_phase(self, phases):
        return np.ones_like(phases)


def test_rejected_acceleration_commits_exact_physical_cycle_bursts(monkeypatch):
    monkeypatch.setenv("V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS", "4")
    monkeypatch.setenv("V10230_HIGH_CYCLE_HEARTBEAT_OPERATIONS", "1000")
    monkeypatch.setenv("V10230_HIGH_CYCLE_EXACT_BURST_INITIAL", "2")
    monkeypatch.setenv("V10230_HIGH_CYCLE_EXACT_BURST_MAX", "4")
    monkeypatch.setenv("V10230_HIGH_CYCLE_EXACT_BURST_GROWTH", "2")
    monkeypatch.setenv("V10230_HIGH_CYCLE_PERIODIC_RETRY_INITIAL", "1000")
    monkeypatch.setenv("V10230_HIGH_CYCLE_PROJECTIVE_RETRY_INITIAL", "1000")

    residual = SimpleNamespace(maximum_relative=1.0, converged=False)
    monkeypatch.setattr(
        high,
        "solve_periodic_state",
        lambda *args, **kwargs: SimpleNamespace(
            converged=False,
            iterations=1,
            map_evaluations=1,
            residual=residual,
            distance_from_initial=1.0,
            failure_reason="forced_nonperiodic",
        ),
    )
    monkeypatch.setattr(
        high,
        "propagate_projective_cycles",
        lambda *args, **kwargs: SimpleNamespace(
            accepted=False,
            cycles_consumed=0.0,
            failure_reason="forced_projective_rejection",
            drift_relative_error=1.0,
            hazard_relative_error=1.0,
            attempts=1,
        ),
    )

    def forbidden_subcycle(*args, **kwargs):
        raise AssertionError("ordinary transient evolution used the subcycle marcher")

    monkeypatch.setattr(
        high._transient,
        "integrate_state_coupled_waveform",
        forbidden_subcycle,
    )

    engine = _Engine()
    result = high.integrate_state_coupled_waveform(
        engine, _Controller(), _Waveform(), 300.0, 10.0
    )

    assert result["fired"] is False
    assert result["coupled_hazard_cycles_consumed"] == 10.0
    assert result["coupled_hazard_partial_return"] is False
    assert result["coupled_hazard_exact_cycle_transient"] is True
    assert result["coupled_hazard_subcycle_marcher_event_only"] is True
    assert sum(
        row["cycles"]
        for row in result["coupled_hazard_modes"]
        if row["mode"] == "exact_cycle_burst"
    ) == 10.0
    assert engine.t == 10.0
    assert np.isclose(engine.mpz.mobile_count, 5.0)
    assert np.isclose(engine.N_em, 5.0)
