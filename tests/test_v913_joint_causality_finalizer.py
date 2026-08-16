import numpy as np
import pandas as pd

from scripts.finalize_v913_joint_fracture_fatigue_causality import barrier


def test_barrier_surface_is_bounded_and_decreases_with_stress():
    row = pd.Series({
        "Tref_K": 300.0,
        "cleave_G00_eV": 2.0,
        "cleave_gT_eV_per_K": 0.0,
        "cleave_sigc0_GPa": 8.0,
        "cleave_sT_GPa_per_K": 0.0,
        "cleave_floor_frac": 0.1,
        "cleave_exp_a": 1.0,
        "cleave_exp_n": 2.0,
    })
    values = barrier(row, "cleave", np.linspace(0, 20, 100), 300.0)
    assert np.all(np.diff(values) <= 0.0)
    assert values[0] <= 2.0
    assert values[-1] >= 0.2
