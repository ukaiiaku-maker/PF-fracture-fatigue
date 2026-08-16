from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("selector", ROOT / "scripts/select_v914_slope_fatigue_loads.py")
selector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(selector)


def test_target_contract_has_six_accelerated_three_explicit_and_overlap():
    modes = [item[2] for item in selector.TARGETS]
    assert sum(mode in {"accelerated", "both"} for mode in modes) == 6
    assert sum(mode in {"explicit", "both"} for mode in modes) == 3
    assert modes.count("both") == 1
    assert len(selector.TARGETS) == 8


def test_log_interpolation_recovers_exact_power_law():
    fraction = np.asarray([0.5, 1.0, 1.5])
    cycles = np.power(10.0, 8.0 - 4.0 * fraction)
    value = selector.interpolate(fraction, cycles, 1.0e4)
    assert math.isclose(value, 1.0, abs_tol=1e-14)
