from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueWaveform
from arrhenius_fracture import persistent_site_explicit_cycle_v1032 as explicit
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    capture_stochastic_state,
)


class FakeMPZ:
    mobile_count = 2.0
    retained_count = 3.0
    escaped_total = 0.0
    emitted_total = 0.0


class FakeEngine:
    explicit_cycle_v1032 = True

    def __init__(self):
        self.explicit_cycle_phase = 0.0
        self.explicit_cycle_index = 0
        self.explicit_cycle_event_count = 0
        self.hazard_threshold_action = 0.6
        self.hazard_action_current = 0.0
        self.hazard_event_index = 0
        self.B = 0.0
        self.mpz = FakeMPZ()
        self.seen = []

    def sigma_tip(self, K):
        return float(K)

    def sigma_back(self):
        return 0.0

    def r_eff(self):
        return 1.0e-6

    def source_geometry(self):
        return {"front_width_m": 10.0e-6}

    def _active_shielding_signed(self):
        return 0.0

    def _integrate_coupled(self, K, T, dt):
        # Five units of physical action per physical cycle at f=1 Hz.
        requested_action = 5.0 * float(dt)
        needed = self.hazard_threshold_action - self.hazard_action_current
        fired = requested_action >= needed
        used = needed / 5.0 if fired else float(dt)
        action = 5.0 * used
        self.hazard_action_current += action
        self.B = self.hazard_action_current / self.hazard_threshold_action
        self.seen.append((float(K), float(getattr(self, "_energy_gate_event_K_override", -1.0))))
        completed = 0.0
        if fired:
            completed = self.hazard_action_current
            self.hazard_action_current = 0.0
            self.B = 0.0
            self.hazard_event_index += 1
        return {
            "fired": fired, "n_fire": int(fired), "dB": action / self.hazard_threshold_action,
            "physical_hazard_action_step": action, "dt_consumed": used,
            "dt_unused": max(float(dt) - used, 0.0), "lambda_c": 5.0,
            "lambda_c_raw": 5.0, "sigma_tip": float(K), "Gc_J": 1.0,
            "plastic": {"dN_emit": used, "dN_escaped": 0.0}, "advance": {},
            "microsteps": 1, "da": 0.0,
            "hazard_action_completed": completed,
        }


def controller(n_phase=8):
    return SimpleNamespace(cfg=FatigueControllerConfig(n_phase=n_phase))


def test_two_events_continue_in_same_physical_cycle(monkeypatch):
    monkeypatch.setattr(explicit, "invalidate_high_cycle_cache", lambda engine, reason: setattr(engine, "invalidated", reason))
    engine = FakeEngine()
    waveform = FatigueWaveform(Kmax=10.0, R=0.1, frequency_Hz=1.0)
    first = explicit.advance_explicit_cycle_remainder(engine, controller(), waveform, 300.0, 1.0)
    first_phase = engine.explicit_cycle_phase
    assert first["fired"] is True
    assert 0.0 < first_phase < 1.0
    assert engine.explicit_cycle_index == 0
    second = explicit.advance_explicit_cycle_remainder(engine, controller(), waveform, 300.0, 1.0)
    assert second["fired"] is True
    assert first_phase < engine.explicit_cycle_phase < 1.0
    assert engine.explicit_cycle_index == 0
    assert engine.invalidated == "explicit_cycle_first_passage_event"
    # Kinetics used the current phase load, while the unchanged energy gate is
    # explicitly told to evaluate the transaction at cycle Kmax.
    assert all(phase_K <= waveform.Kmax for phase_K, _ in engine.seen)
    assert all(event_K == waveform.Kmax for _, event_K in engine.seen)


def test_explicit_selector_and_checkpoint_preserve_phase():
    engine = FakeEngine()
    engine.explicit_cycle_phase = 0.375
    prediction = SimpleNamespace(_v10229_vhcf_engine=engine)
    selected = explicit.select_explicit_cycle_block(None, prediction, 1e9, {"cycles": 1e-6})
    assert selected["cycles"] == 0.625
    state = capture_stochastic_state(engine)
    assert state["explicit_cycle_phase"] == 0.375
    assert state["explicit_cycle_index"] == 0
    assert state["explicit_cycle_event_count"] == 0


def test_one_cycle_boundary_has_exact_cycle_accounting(monkeypatch):
    monkeypatch.setattr(explicit, "invalidate_high_cycle_cache", lambda *args: None)
    engine = FakeEngine()
    engine.hazard_threshold_action = 1.0e9
    waveform = FatigueWaveform(Kmax=10.0, R=0.1, frequency_Hz=1.0)
    result = explicit.advance_explicit_cycle_remainder(engine, controller(16), waveform, 300.0, 1.0)
    assert result["fired"] is False
    assert result["dt_consumed"] == 1.0
    assert engine.explicit_cycle_phase == 0.0
    assert engine.explicit_cycle_index == 1
    assert len(result["explicit_phase_records"]) == 16
