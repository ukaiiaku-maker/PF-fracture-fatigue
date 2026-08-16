import math

import numpy as np
import pandas as pd

from scripts.analyze_v913_joint_paris_slope_physics import (
    conservative_local_slopes,
    exp_floor_from_stress,
)


def material_row() -> pd.Series:
    return pd.Series({
        "Tref_K": 300.0,
        "cleave_G00_eV": 2.4,
        "cleave_gT_eV_per_K": 2e-4,
        "cleave_sigc0_GPa": 8.0,
        "cleave_sT_GPa_per_K": -1e-3,
        "cleave_exp_a": 1.3,
        "cleave_exp_n": 2.2,
        "cleave_floor_frac": 0.08,
    })


def test_exact_exp_floor_stress_derivatives_match_bounded_difference():
    row = material_row(); sigma = 6.2e9; temperature = 700.0
    exact = exp_floor_from_stress(row, sigma, temperature, "cleave")
    step = 1e5
    minus = float(exp_floor_from_stress(row, sigma-step, temperature, "cleave")["G_eV"])
    center = float(exact["G_eV"])
    plus = float(exp_floor_from_stress(row, sigma+step, temperature, "cleave")["G_eV"])
    assert math.isclose(float(exact["dG_dsigma_eV_per_Pa"]), (plus-minus)/(2*step), rel_tol=2e-8)
    assert math.isclose(float(exact["d2G_dsigma2_eV_per_Pa2"]), (plus-2*center+minus)/step**2, rel_tol=2e-4)


def test_temperature_derivative_preserves_analytic_surface_definition():
    row = material_row(); sigma = 7e9; temperature = 800.0; step = 1e-3
    exact = exp_floor_from_stress(row, sigma, temperature, "cleave")
    minus = float(exp_floor_from_stress(row, sigma, temperature-step, "cleave")["G_eV"])
    plus = float(exp_floor_from_stress(row, sigma, temperature+step, "cleave")["G_eV"])
    assert math.isclose(float(exact["dG_dT_eV_per_K"]), (plus-minus)/(2*step), rel_tol=2e-7)


def test_local_loglog_slope_does_not_cross_integration_modes():
    rows=[]
    for mode, scale in (("accelerated", 2.0), ("explicit", 5.0)):
        for k in (10.0, 12.0, 15.0):
            rows.append({"candidate_id":"c", "integration_mode":mode, "deltaK_MPa_sqrt_m":k,
                         "normalized_f":k/10, "developed_da_dN_m_per_cycle":scale*k**4})
    result=conservative_local_slopes(pd.DataFrame(rows))
    assert len(result)==6
    assert np.allclose(result.local_m,4.0,rtol=1e-12,atol=1e-12)
