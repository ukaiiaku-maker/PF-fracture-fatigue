from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_v10_2_22_dbtt_rcurves.py"
SPEC = importlib.util.spec_from_file_location("plot_v10222", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLOTTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTTER)


def _data(rows):
    dtype = [
        ("KJ_Pa_sqrtm", float),
        ("crack_extension_m", float),
        ("da_block_m", float),
        ("n_fire", float),
    ]
    return np.array(rows, dtype=dtype)


def test_event_curve_spans_pre_to_post_crack_extension():
    data = _data(
        [
            (20.0e6, 5.0e-6, 5.0e-6, 1.0),
            (25.0e6, 15.0e-6, 10.0e-6, 1.0),
            (30.0e6, 55.0e-6, 40.0e-6, 1.0),
        ]
    )
    pre, post, resistance = PLOTTER._event_curve(data)
    assert pre == pytest.approx([0.0, 5.0, 15.0])
    assert post == pytest.approx([5.0, 15.0, 55.0])
    assert resistance == pytest.approx([20.0, 25.0, 30.0])
    assert PLOTTER._achieved_extension_um(data) == pytest.approx(55.0)


def test_resistance_queries_require_realized_extension():
    pre = np.array([0.0, 5.0, 15.0])
    post = np.array([5.0, 15.0, 55.0])
    resistance = np.array([20.0, 25.0, 30.0])
    assert PLOTTER._resistance_at_extension(
        pre, post, resistance, 10.0, 55.0
    ) == pytest.approx(25.0)
    assert PLOTTER._resistance_at_extension(
        pre, post, resistance, 25.0, 55.0
    ) == pytest.approx(30.0)
    assert PLOTTER._resistance_at_extension(
        pre, post, resistance, 50.0, 55.0
    ) == pytest.approx(30.0)
    assert math.isnan(
        PLOTTER._resistance_at_extension(
            pre, post, resistance, 60.0, 55.0
        )
    )


def test_short_smoke_does_not_fabricate_10_25_50um_values():
    pre = np.array([0.0, 2.43357])
    post = np.array([2.43357, 8.269426])
    resistance = np.array([28.6, 29.0])
    for target in (10.0, 25.0, 50.0):
        assert math.isnan(
            PLOTTER._resistance_at_extension(
                pre, post, resistance, target, 8.269426
            )
        )
