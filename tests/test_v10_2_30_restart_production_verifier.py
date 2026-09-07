import importlib.util
from pathlib import Path


def verifier_module():
    path = Path(__file__).parents[1] / "scripts" / "verify_v10_2_30_restart_production_complete.py"
    spec = importlib.util.spec_from_file_location("production_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_treats_nan_and_execution_telemetry_as_nonphysical():
    compare = verifier_module().compare_json
    assert compare(float("nan"), float("nan"), "geometry.merge_x_m") == []
    assert compare(1.0, 2.0, "history.audit[0].coupled_hazard_wall_seconds") == []
    assert compare(1.0, 2.0, "history.modes[0].current_map_residual") == []
    assert compare(1.0, 2.0, "history.modes[0].fixed_point_residual") == []


def test_comparison_remains_fail_closed_for_physical_state():
    differences = verifier_module().compare_json(
        {"physical_hazard_action": 0.2, "threshold_action": 0.3},
        {"physical_hazard_action": 0.21, "threshold_action": 0.3},
        "event2",
    )
    assert differences == [{
        "path": "event2.physical_hazard_action", "control": 0.2, "restarted": 0.21,
    }]
