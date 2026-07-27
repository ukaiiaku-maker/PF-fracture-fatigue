from __future__ import annotations

from types import SimpleNamespace

import pytest

from arrhenius_fracture.config import EV_TO_J
from arrhenius_fracture.hazard_energy_gate_v10230 import (
    hazard_dissipation_density_J_per_m2,
)


class _DynamicEngine:
    def __init__(self):
        self.f = SimpleNamespace(m_hits=3.0)
        self.b = 2.74e-10
        self.state_shift_eV = 0.0

    def lambda_cleave(self, sigma, T):
        barrier_eV = (
            3.0
            + 5.0e-4 * (float(T) - 300.0)
            - 1.0e-10 * float(sigma)
            + self.state_shift_eV
        )
        barrier_eV = max(barrier_eV, 1.0e-6)
        return 1.0, 1.0, barrier_eV * EV_TO_J


def test_Gamma_haz_is_recomputed_from_temperature_stress_and_state():
    engine = _DynamicEngine()
    G_reference, _ = hazard_dissipation_density_J_per_m2(
        engine, 300.0, 1.0e9, 1.0
    )
    G_hot, _ = hazard_dissipation_density_J_per_m2(
        engine, 900.0, 1.0e9, 1.0
    )
    G_stressed, _ = hazard_dissipation_density_J_per_m2(
        engine, 300.0, 5.0e9, 1.0
    )
    engine.state_shift_eV = -0.4
    G_evolved, _ = hazard_dissipation_density_J_per_m2(
        engine, 300.0, 1.0e9, 1.0
    )

    assert G_hot > G_reference
    assert G_stressed < G_reference
    assert G_evolved < G_reference
    assert len({G_reference, G_hot, G_stressed, G_evolved}) == 4


def test_relative_plane_factor_scales_dynamic_dissipation_linearly():
    engine = _DynamicEngine()
    G1, _ = hazard_dissipation_density_J_per_m2(
        engine, 700.0, 2.0e9, 1.0
    )
    G13, _ = hazard_dissipation_density_J_per_m2(
        engine, 700.0, 2.0e9, 1.3
    )
    assert G13 == pytest.approx(1.3 * G1)
