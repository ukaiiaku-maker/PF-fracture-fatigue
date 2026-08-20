from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_v914_v7_causal_physics import (
    cleavage_ceiling,
    cleavage_components,
    exp_floor_limit_eV,
    geometry_audit,
)


class Surface:
    floor_max_fraction = 0.95
    floor_min_eV = 1.0e-4
    floor_fraction = 0.01

    def zero_stress_eV(self, _temperature):
        return 2.0

    def barrier_eV(self, stress, _temperature):
        return np.asarray(1.0 - 1.0e-12 * np.asarray(stress))


def test_exp_floor_limit_uses_active_max_min_contract():
    assert exp_floor_limit_eV(Surface(), 300.0) == 0.02


def test_counterfactual_radius_and_shielding_do_not_mutate_state():
    state = SimpleNamespace(
        c=SimpleNamespace(
            b_m=1.0e-10, cleavage_nu0_s=1.0e12,
            cleavage_hits=3.0, cleavage_correlation_time_s=1.0e-6,
        ),
        p=SimpleNamespace(cleavage=Surface()),
        tip_radius_m=lambda: 2.0e-6,
        K_shield_MPa_sqrt_m=lambda: 3.0,
    )
    result = cleavage_components(
        state, 10.0, 300.0, radius_m=1.0e-6, shielding_MPa_sqrt_m=0.0
    )
    assert result["radius_m"] == 1.0e-6
    assert result["shielding_MPa_sqrt_m"] == 0.0
    assert state.tip_radius_m() == 2.0e-6
    assert state.K_shield_MPa_sqrt_m() == 3.0


def test_three_hit_ceiling_uses_production_gammainc_definition():
    candidate = SimpleNamespace(cleavage=Surface())
    physics = SimpleNamespace(
        cleavage_nu0_s=1.0e12,
        cleavage_hits=3.0,
        cleavage_correlation_time_s=1.0e-6,
    )
    result = cleavage_ceiling(candidate, physics, 300.0, 1000.0)
    raw = 1.0e12 * math.exp(-0.02 / (8.617333262145e-5 * 300.0))
    expected = gammainc(3.0, raw * 1.0e-6) / 1.0e-6
    assert math.isclose(result["Gamma_raw_max_s"], raw, rel_tol=2.0e-10)
    assert math.isclose(result["Gamma_c_max_s"], expected, rel_tol=2.0e-10)
    assert result["maximum_action_per_cycle"] == expected / 1000.0


def test_geometry_audit_includes_source_zone_tip_weight():
    candidate = SimpleNamespace(
        c_blunt=2.0,
        emission=SimpleNamespace(characteristic_stress_Pa=lambda _T: 1.0e9),
    )
    physics = SimpleNamespace(
        mpz_length_m=4.0,
        n_bins=4,
        blunting_length_m=1.0,
        b_m=0.5,
        source_zone_length_m=2.0,
        activation_to_line_content_per_system=(1.0, 2.0),
        blunting_slip_fraction=1.0,
        G_Pa=1.0,
        nu=0.25,
    )
    result = geometry_audit(candidate, physics, [1.0], 300.0)
    weight = 0.5 * (math.exp(-0.5) + math.exp(-1.5))
    assert result["source_zone_bin_count"] == 2
    assert math.isclose(result["source_zone_mean_tip_weight"], weight)
    assert np.allclose(result["one_activation_tip_radius_increment_m"], [weight, 2.0 * weight])
