from __future__ import annotations

import numpy as np

from arrhenius_fracture.material_manifest import ExpFloorBarrier
from arrhenius_fracture.thermodynamic_net_slip_v1041 import (
    audit_payload, detailed_balance_rate_s,
    install_detailed_balance_net_slip, restore_detailed_balance_net_slip,
)


def _surface() -> ExpFloorBarrier:
    return ExpFloorBarrier(
        G00_eV=0.6, gT_eV_per_K=0.0, sigc0_Pa=2e9,
        sT_Pa_per_K=0.0, alpha=0.5, exponent=1.2,
        floor_fraction=0.02, floor_min_eV=1e-4,
        floor_max_fraction=0.95, Tref_K=300.0,
        attempt_frequency_s=1e12,
    )


def test_detailed_balance_current_surface_is_zero_and_monotone():
    surface = _surface()
    stress = np.array([0.0, 1e8, 5e8, 1e9])
    rate = detailed_balance_rate_s(surface, stress, 1000.0)
    assert rate[0] == 0.0
    assert np.all(rate[1:] > 0.0)
    assert np.all(np.diff(rate) >= 0.0)


def test_overlay_is_reversible_and_audited():
    surface = _surface()
    original_bound = surface.rate(np.array([1e8]), 1000.0)
    original = install_detailed_balance_net_slip()
    try:
        assert surface.rate(np.array([0.0]), 1000.0)[0] == 0.0
    finally:
        restore_detailed_balance_net_slip(original)
    assert np.array_equal(surface.rate(np.array([1e8]), 1000.0), original_bound)
    assert audit_payload()["new_fitted_parameters"] == 0
