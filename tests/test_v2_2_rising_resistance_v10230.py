import math

import pandas as pd

from scripts.run_v2_2_1d_rising_resistance_v10230 import classify


def frame(values, candidate="x"):
    return pd.DataFrame({
        "candidate_id": [candidate] * len(values),
        "K_reinit_MPa_sqrt_m": values,
        "cumulative_advance_um": [5.0 * (i + 1) for i in range(len(values))],
        "constitutive_ceiling_occupied": False, "sigma_cap_active": False,
        "dN_cap_active": False, "N_sat_active": False,
        "conservation_relative": 0.0,
    })


def test_preregistered_strong_gate_and_opening_ablation():
    result = classify(frame([10, 11, 12, 13, 14, 15, 16, 17, 18, 20]), frame([10] * 10))
    assert result["classification"] == "1D_STRONG_RISING_RESISTANCE"
    assert math.isclose(result["normalized_rise"], 1.0)


def test_flat_gate_is_not_promoted():
    assert classify(frame([10] * 10), frame([10] * 10))["classification"] == "1D_EFFECTIVE_RESISTANCE_FLAT"
