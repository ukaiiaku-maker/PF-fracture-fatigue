from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.fatigue_reintegration_v1020 import (
    is_v10_moving_engine,
    make_cycle_step_dispatch,
    predict_one_cycle_v1020,
)


class DummyMPZ:
    def __init__(self):
        self.n_systems = 2
        self.available_sites = np.array([2.0, 3.0])
        self._anisotropic_drive_factors = np.array([0.5, 1.5])
        self.mobile_count = 1.0
        self.retained_count = 2.0
        self.emitted_total = 3.0
        self.escaped_total = 4.0
        self.recovered_total = 5.0

    @staticmethod
    def emission_rate_per_site(stress, temperature):
        return max(float(stress), 0.0) / 1.0e9


class DummyFront:
    kinetic_tip_cell_active = True
    stochastic_hazard_threshold_active = True
    stochastic_avalanche_length_active = True
    _audit_records = []

    def __init__(self):
        self.mpz = DummyMPZ()
        self.B = 0.2
        self.N_em = 2.0
        self.avalanche_last_completed_advance_m = 7.5e-6
        self.avalanche_last_completed_factor = 1.5
        self.avalanche_event_advance_m = 6.0e-6
        self.avalanche_event_length_factor = 1.2
        self.native_called = False

    @staticmethod
    def sigma_tip(K):
        return float(K) * 100.0

    @staticmethod
    def lambda_cleave(stress, temperature):
        return float(stress) * 1.0e-12, 0.0, 1.0

    @staticmethod
    def r_eff():
        return 1.0e-6

    @staticmethod
    def cleavage_diagnostics(stress, temperature):
        return {"G_cleave_eff_eV": 1.0, "G_cleave_raw_eV": 1.1}

    def cycle_step_waveform(
        self, controller, waveform, temperature, requested_cycles=None, force_cycles=None
    ):
        self.native_called = True
        self.mpz.mobile_count += 0.4
        self.mpz.retained_count += 0.3
        self.mpz.emitted_total += 1.0
        self.mpz.escaped_total += 0.2
        self.mpz.recovered_total += 0.1
        return {
            "cycles": 10.0,
            "dB": 0.1,
            "mu_cleave_pred": 0.01,
            "fired": True,
            "n_fire": 1,
            "v_crack": 1.0e-6,
            "lambda_c": 1.0,
            "lambda_e": 2.0,
        }


class DummyWaveform:
    Kmax = 2.0e6
    R = 0.1
    frequency_Hz = 1000.0
    period_s = 1.0e-3
    DeltaK = 1.8e6

    @staticmethod
    def K_phase(phase):
        return 1.0e6 * (1.0 + 0.5 * np.cos(phase))


class DummyController:
    cfg = SimpleNamespace(
        block_cycles=100.0,
        max_block_cycles=1.0e6,
        min_block_cycles=1.0e-6,
        adaptive_cycles=True,
        target_dB=0.2,
        target_dN_emit=0.25,
    )

    @staticmethod
    def _phases():
        return np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)


def test_predictor_uses_current_source_budget(monkeypatch):
    monkeypatch.setattr(
        "arrhenius_fracture.fatigue_reintegration_v1020._campaign_backstress",
        lambda state: (np.zeros(2), np.zeros(2), np.zeros(2)),
    )
    prediction = predict_one_cycle_v1020(
        DummyFront(), DummyWaveform(), 700.0, DummyController()
    )
    assert prediction.mu_emit > 0.0
    assert prediction.mu_cleave > 0.0
    assert prediction.store_per_cycle == prediction.mu_emit
    assert prediction.max_sigma_tip > prediction.avg_sigma_tip


def test_cycle_dispatch_uses_native_moving_mpz_path():
    front = DummyFront()

    def legacy(*args, **kwargs):
        raise AssertionError("legacy scalar fatigue path was used")

    dispatch = make_cycle_step_dispatch(legacy)
    result = dispatch(
        DummyController(), front, DummyWaveform(), 700.0, force_cycles=10.0
    )
    assert front.native_called
    assert result["fatigue_native_moving_mpz_dispatch"] is True
    assert result["dN_emit_block"] == 1.0
    assert np.isclose(result["dN_store_block"], 0.3)
    assert np.isclose(result["dN_mobile_block"], 0.4)
    assert result["avalanche_event_advance_m"] == 7.5e-6


def test_nonmoving_engine_is_not_dispatched():
    assert not is_v10_moving_engine(SimpleNamespace())
