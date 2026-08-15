import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts/analyze_v914_endurance_knee_rerank.py"
SPEC = importlib.util.spec_from_file_location("rerank", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def curve(rates):
    return pd.DataFrame({
        "deltaK_MPa_sqrt_m": np.arange(len(rates), dtype=float),
        "da_dN_m_per_cycle": rates,
    })


def test_terminal_plateau_is_not_reported_as_slope_recovery():
    metrics = MODULE.curve_metrics(curve(10.0 ** np.array([-12, -8, -5, -4.9, -4.85, -4.84])))
    assert metrics["R_recovery"] <= 1.0


def test_localized_flat_window_with_recovered_slope_is_detected():
    metrics = MODULE.curve_metrics(curve(10.0 ** np.array([-12, -9, -8.9, -6, -3])))
    assert metrics["knee_quality"] > 1.0
    assert metrics["R_recovery"] > 1.0
