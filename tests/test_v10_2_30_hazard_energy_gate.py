from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.config import EV_TO_J
from arrhenius_fracture.hazard_energy_gate_v10230 import (
    HazardEnergyGateContext,
    HazardEnergyGatedPersistentSiteCyclicTipEngine,
    canonical_gamma_rel,
    gate_fraction_from_context,
    hazard_K_from_event_K,
    hazard_dissipation_density_J_per_m2,
    probe_to_event_energy_scale,
)
from arrhenius_fracture.hazard_energy_observed_engine_v10230 import (
    ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from arrhenius_fracture import hazard_energy_observer_v10230 as observer
from arrhenius_fracture import stochastic_avalanche_tip as avalanche


ROOT = Path(__file__).resolve().parents[1]


class _FakeFrontConfig:
    m_hits = 3.0


class _FakeEngine:
    f = _FakeFrontConfig()
    b = 2.0e-10

    def lambda_cleave(self, sigma, T):
        return 7.0, 8.0, 2.0 * EV_TO_J


def test_hazard_dissipation_density_uses_active_production_terms():
    gamma = 1.3
    Gamma, deltaG = hazard_dissipation_density_J_per_m2(
        _FakeEngine(),
        T_K=900.0,
        sigma_cleave_eff_Pa=5.0e9,
        gamma_rel=gamma,
    )
    expected = gamma * 3.0 * (2.0 * EV_TO_J) / (2.0e-10) ** 2
    assert deltaG == pytest.approx(2.0 * EV_TO_J)
    assert Gamma == pytest.approx(expected)


def test_probe_to_event_energy_scaling_is_quadratic():
    assert probe_to_event_energy_scale(20.0, 10.0) == pytest.approx(4.0)
    assert probe_to_event_energy_scale(5.0, 10.0) == pytest.approx(0.25)
    assert probe_to_event_energy_scale(0.0, 0.0) == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="zero FEM probe K"):
        probe_to_event_energy_scale(1.0, 0.0)


def test_gate_fraction_limits_event_reward_by_available_work():
    engine = _FakeEngine()
    Gamma, _ = hazard_dissipation_density_J_per_m2(
        engine, 900.0, 5.0e9, 1.0
    )
    context = HazardEnergyGateContext(
        J_probe_J_per_m2=0.5 * Gamma,
        K_probe_Pa_sqrt_m=10.0,
        K_event_Pa_sqrt_m=10.0,
        gamma_rel=1.0,
        loading_mode="monotonic",
    )
    result = gate_fraction_from_context(engine, context, 900.0, 5.0e9)
    assert result["gate_fraction"] == pytest.approx(0.5)
    assert result["Gamma_haz_J_per_m2"] == pytest.approx(Gamma)
    assert result["J_event_scaled_J_per_m2"] == pytest.approx(0.5 * Gamma)


def test_existing_angular_factor_scales_hazard_and_dissipation_consistently():
    gamma = 4.0
    assert hazard_K_from_event_K(12.0, gamma) == pytest.approx(6.0)
    candidate = {"gamma": gamma}
    assert canonical_gamma_rel(candidate) == pytest.approx(gamma)
    assert candidate["gamma_rel"] == pytest.approx(gamma)


def _bare_observed_engine() -> ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine:
    engine = object.__new__(
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    engine.hazard_energy_gate_context = HazardEnergyGateContext(
        J_probe_J_per_m2=10.0,
        K_probe_Pa_sqrt_m=5.0,
        K_event_Pa_sqrt_m=5.0,
        gamma_rel=4.0,
        loading_mode="monotonic",
    )
    engine.hazard_energy_last_sigma_physical_Pa = 0.0
    engine.hazard_energy_last_sigma_scaled_Pa = 0.0
    return engine


def test_observed_engine_scales_only_cleavage_hazard_stress(monkeypatch):
    engine = _bare_observed_engine()

    def base_lambda(self, sigma, T):
        return float(sigma), float(sigma) + 1.0, 2.0 * EV_TO_J

    monkeypatch.setattr(
        HazardEnergyGatedPersistentSiteCyclicTipEngine,
        "lambda_cleave",
        base_lambda,
        raising=False,
    )
    effective, raw, barrier = (
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine.lambda_cleave(
            engine, 100.0, 900.0
        )
    )
    assert effective == pytest.approx(50.0)
    assert raw == pytest.approx(51.0)
    assert barrier == pytest.approx(2.0 * EV_TO_J)
    assert engine.hazard_energy_last_sigma_physical_Pa == pytest.approx(100.0)
    assert engine.hazard_energy_last_sigma_scaled_Pa == pytest.approx(50.0)


def test_zero_positive_J_suppresses_hazard_without_athermal_threshold(monkeypatch):
    engine = _bare_observed_engine()
    engine.hazard_energy_gate_context = HazardEnergyGateContext(
        J_probe_J_per_m2=0.0,
        K_probe_Pa_sqrt_m=5.0,
        K_event_Pa_sqrt_m=5.0,
        gamma_rel=1.0,
        loading_mode="monotonic",
    )

    monkeypatch.setattr(
        HazardEnergyGatedPersistentSiteCyclicTipEngine,
        "lambda_cleave",
        lambda self, sigma, T: (9.0, 10.0, 3.0 * EV_TO_J),
        raising=False,
    )
    effective, raw, barrier = (
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine.lambda_cleave(
            engine, 100.0, 900.0
        )
    )
    assert effective == 0.0
    assert raw == 0.0
    assert barrier == pytest.approx(3.0 * EV_TO_J)


def test_observer_normalizes_gamma_and_tracks_negative_to_positive_J(monkeypatch):
    observer.restore_observer()
    results = iter(
        [
            (0.01, 5.0, {"J_signed": -0.01}),
            (16.0, 7.0, {"J_signed": 16.0}),
        ]
    )

    def fake_J(*args, **kwargs):
        return next(results)

    monkeypatch.setattr(observer.j_integral, "compute_J_integral", fake_J)
    observer.install_observer()
    try:
        sigma = np.array([[1.0, 0.0], [0.0, 2.0]])
        selected, _all = observer.crystal.cleave_direction_competition(
            sigma,
            theta_deg=15.0,
            forward=np.array([1.0, 0.0]),
            min_forward=0.2,
            gamma_aniso=0.3,
            branch_ratio=0.9,
        )
        assert selected
        assert "gamma_rel" in selected[0]

        observer.j_integral.compute_J_integral()
        startup = observer.current_observation()
        assert startup.gamma_rel == pytest.approx(selected[0]["gamma_rel"])
        assert startup.J_sign_reference == 1.0
        assert startup.J_probe_J_per_m2 == 0.0
        assert startup.K_probe_Pa_sqrt_m == 0.0
        assert startup.first_nonzero_sign_latch_used is False

        observer.j_integral.compute_J_integral()
        loaded = observer.current_observation()
        assert loaded.J_sign_reference == 1.0
        assert loaded.J_probe_J_per_m2 == pytest.approx(16.0)
        assert loaded.K_probe_Pa_sqrt_m == pytest.approx(7.0)
        assert loaded.J_sign_convention == (
            "positive_raw_signed_J_is_forward_configurational_work"
        )
        assert loaded.first_nonzero_sign_latch_used is False
    finally:
        observer.restore_observer()


def test_integrated_work_audit_closes_for_completed_event(monkeypatch):
    engine = object.__new__(
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    engine.hazard_energy_available_event_accum_J_per_m = 0.0
    engine.hazard_energy_dissipated_event_accum_J_per_m = 0.0
    engine.hazard_energy_gate_event_history = [{}]
    engine.hazard_energy_gate_last = {}
    avalanche._PENDING_GEOMETRY_EVENTS.clear()
    avalanche._PENDING_GEOMETRY_EVENTS.append({"hazard_energy_gate": {}})

    monkeypatch.setattr(
        HazardEnergyGatedPersistentSiteCyclicTipEngine,
        "_integrate_coupled",
        lambda self, *args, **kwargs: {
            "fired": True,
            "dB": 1.0,
            "hazard_energy_gate_proposed_checkpoint_m": 2.0,
            "hazard_energy_gate_accepted_step_m": 1.0,
            "J_event_scaled_J_per_m2": 4.0,
            "Gamma_haz_J_per_m2": 4.0,
        },
    )
    result = (
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine._integrate_coupled(
            engine
        )
    )
    assert result["energy_available_integrated_J_per_m"] == pytest.approx(8.0)
    assert result["energy_dissipated_integrated_J_per_m"] == pytest.approx(4.0)
    assert result["integrated_energy_balance_pass"] is True
    assert avalanche._PENDING_GEOMETRY_EVENTS[-1]["hazard_energy_gate"][
        "energy_margin_integrated_J_per_m"
    ] == pytest.approx(4.0)
    avalanche._PENDING_GEOMETRY_EVENTS.clear()


def test_integrated_work_violation_fails_closed(monkeypatch):
    engine = object.__new__(
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    engine.hazard_energy_available_event_accum_J_per_m = 0.0
    engine.hazard_energy_dissipated_event_accum_J_per_m = 0.0
    engine.hazard_energy_gate_event_history = [{}]
    engine.hazard_energy_gate_last = {}
    avalanche._PENDING_GEOMETRY_EVENTS.clear()

    monkeypatch.setattr(
        HazardEnergyGatedPersistentSiteCyclicTipEngine,
        "_integrate_coupled",
        lambda self, *args, **kwargs: {
            "fired": True,
            "dB": 1.0,
            "hazard_energy_gate_proposed_checkpoint_m": 1.0,
            "hazard_energy_gate_accepted_step_m": 2.0,
            "J_event_scaled_J_per_m2": 1.0,
            "Gamma_haz_J_per_m2": 1.0,
        },
    )
    with pytest.raises(RuntimeError, match="integrated work balance"):
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine._integrate_coupled(
            engine
        )


def test_geometry_veto_never_performs_partial_scalar_rollback():
    engine = object.__new__(
        ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    with pytest.raises(RuntimeError, match="Exact rollback requires replay"):
        engine.restore_geometry_veto(1)


def test_v10230_entry_contract_has_no_athermal_Gc():
    source = (
        ROOT
        / "arrhenius_fracture"
        / "sharp_front_v10_2_30_hazard_energy_gated.py"
    ).read_text()
    assert "absolute_athermal_Gc" in source
    assert '"absolute_athermal_Gc": False' in source
    assert "Gamma_haz=gamma_rel*m*DeltaG_cleave_eff(T,sigma)/b^2" in source
    assert '"fixed_DeltaK_energy_scaling": "(K_event/K_probe)^2"' in source
    assert "install_hazard_energy_backend_audit" in source
    assert "install_observer" in source
