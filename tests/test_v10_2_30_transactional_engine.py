from types import MethodType, SimpleNamespace

from arrhenius_fracture import persistent_site_cyclic_energy_gated_corrected_v10230 as corrected
from arrhenius_fracture import stochastic_avalanche_tip


def test_continuum_comparison_is_diagnostic_only(monkeypatch):
    def fake_parent(self, K, T, dt, stress_override=None, lambda_override=None):
        self.energy_gate_last_continuum = {
            "energy_gate_continuum_open": False,
            "hazard_resistance_J_per_m2": 2.0,
            "continuum_driving_J_per_m2": 1.0,
        }
        return {
            "fired": False,
            "hazard_energy_gate_continuum_affects_hazard": False,
        }

    monkeypatch.setattr(
        corrected._base.HazardEnergyGatedPersistentSiteCyclicTipEngine,
        "_integrate_coupled",
        fake_parent,
    )

    engine = object.__new__(
        corrected.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    engine.energy_gate_last_continuum = {}
    engine._energy_gate_pending = None
    result = engine._integrate_coupled(4.0, 300.0, 1.0)

    assert result["hazard_energy_gate_continuum_open"] is False
    assert result["hazard_energy_gate_continuum_affects_hazard"] is False


def test_completed_event_barrier_is_reevaluated_at_Kmax(monkeypatch):
    stochastic_avalanche_tip.clear_pending_geometry_events()

    def fake_parent(self, K, T, dt, stress_override=None, lambda_override=None):
        self.energy_gate_last_continuum = {"energy_gate_continuum_open": True}
        descriptor = {"energy_gate_engine_id": self._engine_id}
        self._energy_gate_pending = {"descriptor": descriptor}
        stochastic_avalanche_tip._PENDING_GEOMETRY_EVENTS.append(descriptor)
        return {"fired": True}

    monkeypatch.setattr(
        corrected._base.HazardEnergyGatedPersistentSiteCyclicTipEngine,
        "_integrate_coupled",
        fake_parent,
    )

    engine = object.__new__(
        corrected.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    engine._engine_id = 17
    engine.energy_gate_last_continuum = {}
    engine._energy_gate_pending = None
    engine.sigma_tip = MethodType(lambda self, K: 2.0 * K, engine)
    engine.lambda_cleave = MethodType(
        lambda self, sigma, T: (1.0, 1.0, sigma + T), engine
    )

    result = engine._integrate_coupled(
        5.0,
        300.0,
        1.0,
        stress_override=3.0,
        lambda_override=4.0,
    )
    descriptor = engine._energy_gate_pending["descriptor"]
    assert descriptor["event_K_Pa_sqrt_m"] == 5.0
    assert descriptor["event_sigma_tip_Pa"] == 10.0
    assert descriptor["hazard_barrier_J"] == 310.0
    assert result["hazard_barrier_J"] == 310.0


def test_zero_length_mesh_arrest_consumes_attempt(monkeypatch):
    result_ref = {"fired": True, "n_fire": 1}
    engine = object.__new__(
        corrected.CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    engine._engine_id = 9
    engine.n_adv = 4
    engine.energy_gate_zero_length_attempt_count = 0
    engine._energy_gate_pending = {
        "descriptor": {"energy_gate_result_ref": result_ref},
        "n_adv_before": 3,
        "proposal_m": 5.0e-6,
    }
    monkeypatch.setattr(
        corrected._gate,
        "_LAST_BACKEND",
        SimpleNamespace(
            advance_log=[
                {
                    "inserted": False,
                    "arrest_reason": "no_mesh_resolved_admissible_increment",
                    "committed_event_length_m": 0.0,
                }
            ]
        ),
    )

    engine.restore_geometry_veto(1)

    assert engine._energy_gate_pending is None
    assert engine.n_adv == 3
    assert engine.energy_gate_zero_length_attempt_count == 1
    assert result_ref["fired"] is False
    assert result_ref["n_fire"] == 0
    assert result_ref["hazard_energy_gate_attempt_consumed"] is True
    assert result_ref["avalanche_event_advance_m"] == 0.0
