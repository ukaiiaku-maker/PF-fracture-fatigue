import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_v10_2_29_developed_fatigue_growth.py"
SPEC = importlib.util.spec_from_file_location("developed_growth_v10229", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _rows():
    dtype = [
        ("step", float),
        ("crack_extension_m", float),
        ("da_block_m", float),
        ("n_fire", float),
        ("fatigue_cycles", float),
        ("KJ_Pa_sqrtm", float),
        ("sigma_back_Pa", float),
    ]
    rows = np.zeros(6, dtype=dtype)
    rows["step"] = np.arange(1, 7)
    rows["fatigue_cycles"] = [10, 10, 20, 20, 30, 30]
    rows["KJ_Pa_sqrtm"] = 1.0e6
    rows["sigma_back_Pa"] = np.arange(6) * 1.0e6
    rows["n_fire"][[1, 3, 5]] = 1
    rows["da_block_m"][[1, 3, 5]] = [5.0e-6, 20.0e-6, 20.0e-6]
    rows["crack_extension_m"] = [0, 5.0e-6, 5.0e-6, 25.0e-6, 25.0e-6, 45.0e-6]
    return rows


def _control():
    return {
        "target_Kmax_MPa_sqrt_m": 10.0,
        "target_deltaK_MPa_sqrt_m": 9.0,
    }


def test_initiation_and_burnin_are_excluded():
    events, measurements, summary = MODULE.extract(
        _rows(),
        _control(),
        developed_start_um=20.0,
        developed_end_um=60.0,
    )
    assert len(events) == 3
    assert events[0]["stage"] == "initiation"
    assert not events[0]["measurement_eligible"]
    assert not events[1]["measurement_eligible"]
    assert events[2]["measurement_eligible"]
    assert len(measurements) == 1
    assert np.isclose(measurements[0]["da_dN_m_per_cycle"], 20.0e-6 / 60.0)
    assert summary["status"] == "developed_measurements"


def test_no_event_is_right_censored():
    rows = _rows()
    rows["n_fire"] = 0
    rows["da_block_m"] = 0
    rows["crack_extension_m"] = 0
    events, measurements, summary = MODULE.extract(
        rows,
        _control(),
        developed_start_um=20.0,
        developed_end_um=None,
    )
    assert events == []
    assert measurements == []
    assert summary["status"] == "right_censored_no_event"
