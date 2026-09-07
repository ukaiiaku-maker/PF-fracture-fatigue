import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (Path(__file__).parents[1] / "scripts"
          / "verify_v10_2_30_near_threshold_production_gate.py")
SPEC = importlib.util.spec_from_file_location("near_threshold_production_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_immutable_near_threshold_checkpoint_fixtures_are_complete():
    for name, expected in MODULE.CASES.items():
        fixture = MODULE.validate_fixture(name)
        outer = fixture["outer"]
        kinetic = fixture["kinetic"]
        arrays = fixture["arrays"]
        assert outer["cycles_total"] == expected["start_cycles"]
        assert outer["geometry"]["committed_event_count"] == expected["start_events"]
        assert kinetic["stochastic"]["B"] == 0.9999999998999998
        assert kinetic["stochastic"]["rng_state"]
        assert arrays["kinetic_active_vector"].shape == (2882,)
        assert np.isfinite(arrays["kinetic_active_vector"]).all()


def test_gate_contract_bounds_locator_and_declares_cycle_tolerance():
    assert MODULE.MAX_LOCATOR_EVALUATIONS == 100
    assert MODULE.LOCALIZATION_ABS_CYCLE_TOL == 1.0e-6
    assert set(MODULE.CASES) == {"dbtt", "peak"}
