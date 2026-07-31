from __future__ import annotations

import numpy as np

from arrhenius_fracture.emission_derived_plasticity import (
    EmissionDerivedPeierlsTaylorConfig,
    EmissionDerivedPeierlsTaylorModel,
    ExpFloorSurface,
)
from arrhenius_fracture.thermodynamic_net_slip_v1041 import (
    audit_payload,
    detailed_balance_rate_s,
    install_detailed_balance_net_slip,
    restore_detailed_balance_net_slip,
)


def _surface(attempt_frequency_s: float = 1.0e12) -> ExpFloorSurface:
    return ExpFloorSurface(
        G00_eV=0.6,
        gT_eV_per_K=0.0,
        sigc0_Pa=2.0e9,
        sT_Pa_per_K=0.0,
        alpha=0.5,
        exponent=1.2,
        floor_fraction=0.02,
        floor_min_eV=1.0e-4,
        floor_max_fraction=0.95,
        Tref_K=300.0,
        attempt_frequency_s=attempt_frequency_s,
    )


def test_detailed_balance_rate_is_exactly_zero_at_zero_stress():
    surface = _surface()
    for temperature in (300.0, 1000.0, 1300.0):
        rate = detailed_balance_rate_s(surface, np.zeros(4), temperature)
        assert np.array_equal(rate, np.zeros(4))


def test_detailed_balance_rate_is_positive_and_monotone_under_drive():
    surface = _surface()
    stress = np.array([0.0, 1.0e8, 5.0e8, 1.0e9])
    rate = detailed_balance_rate_s(surface, stress, 1000.0)
    one_way = surface.rate_s(stress, 1000.0)

    assert rate[0] == 0.0
    assert np.all(rate[1:] > 0.0)
    assert np.all(np.diff(rate) >= 0.0)
    assert np.all(rate <= one_way * (1.0 + 1.0e-14))


def test_installed_model_has_zero_net_orowan_rate_at_zero_stress():
    peierls = _surface(1.0e12)
    taylor = _surface(1.0e11)
    cfg = EmissionDerivedPeierlsTaylorConfig(
        peierls=peierls,
        taylor=taylor,
        taylor_corr_rho_c_m2=1.0e14,
        taylor_renewal_time_s=1.0e-9,
        taylor_m_exponent=1.0,
        taylor_m_scale=2.0,
        taylor_m_cap=float("inf"),
        mobile_fraction=0.01,
        mobile_saturation_density_m2=1.0e14,
        mobile_density_floor_m2=5.0e12,
        jump_fraction=1.0,
        jump_length_min_m=2.5e-10,
        taylor_phi_max=20.0,
    )
    model = EmissionDerivedPeierlsTaylorModel(cfg)
    original = install_detailed_balance_net_slip()
    try:
        rates = model.rates(
            np.zeros(3),
            np.full(3, 5.0e12),
            1000.0,
            2.74e-10,
        )
    finally:
        restore_detailed_balance_net_slip(original)

    assert np.array_equal(rates["peierls_rate_s"], np.zeros(3))
    assert np.array_equal(rates["taylor_single_hit_rate_s"], np.zeros(3))
    assert np.array_equal(rates["series_rate_s"], np.zeros(3))
    assert np.array_equal(rates["equivalent_plastic_rate_s"], np.zeros(3))


def test_detailed_balance_audit_rejects_v1040_output_compatibility():
    payload = audit_payload()
    assert payload["zero_stress_net_plastic_rate_exactly_zero"] is True
    assert payload["one_way_arrhenius_rate_used_as_net_slip"] is False
    assert payload["new_fitted_parameters"] == 0
    assert payload["v10_4_0_outputs_physics_compatible"] is False
