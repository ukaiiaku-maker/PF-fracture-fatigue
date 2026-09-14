import math

import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import load_rows
from scripts.run_v2_2_1d_rising_resistance_v10230 import classify, prepare_engine
import scripts.complete_corrected_thermodynamic_joint_search_v10230 as v2


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


def test_canonical_byte_row_is_explicitly_adapted_to_production_numeric_types():
    v2.ROWS = v2.source_rows()
    v2.ACTIVE = v2.active_stresses(v2.ROWS)
    row = next(iter(load_rows().values()))
    engine, _cleavage, _emission, parent = prepare_engine(row, "FULL_PRODUCTION_STATE")
    assert engine.f.da == 5.0e-6
    assert engine.f.m_hits == float(parent["physics__cleavage_hits"])
    assert engine.f.tau_c == float(parent["physics__cleavage_correlation_time_s"])
