from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_v10_2_29_fatigue_growth.py"
spec = importlib.util.spec_from_file_location("extract_v10229", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_extracts_consumed_cycle_event_rate_and_weighted_K():
    dtype = [
        ("step", float),
        ("KJ_Pa_sqrtm", float),
        ("crack_extension_m", float),
        ("da_block_m", float),
        ("n_fire", float),
        ("fatigue_cycles", float),
    ]
    rows = np.array(
        [
            (1, 10.0e6, 0.0, 0.0, 0.0, 100.0),
            (2, 12.0e6, 0.0, 0.0, 0.0, 200.0),
            (3, 14.0e6, 5.0e-6, 5.0e-6, 1.0, 50.0),
            (4, 16.0e6, 5.0e-6, 0.0, 0.0, 150.0),
            (5, 18.0e6, 10.0e-6, 5.0e-6, 1.0, 50.0),
        ],
        dtype=dtype,
    )
    events = module.extract_events(rows, 0.1)
    assert len(events) == 2
    first = events[0]
    assert first["cycles_between_events"] == 350.0
    assert first["event_advance_m"] == 5.0e-6
    assert np.isclose(first["da_dN_raw_m_per_cycle"], 5.0e-6 / 350.0)
    expected_K = (10.0e6 * 100.0 + 12.0e6 * 200.0 + 14.0e6 * 50.0) / 350.0
    assert np.isclose(first["Kmax_cycle_weighted_Pa_sqrt_m"], expected_K)
    assert np.isclose(first["DeltaK_event_pre_Pa_sqrt_m"], 0.9 * 14.0e6)
    second = events[1]
    assert second["cycles_between_events"] == 200.0
    assert np.isclose(second["a_mid_m"], 7.5e-6)
