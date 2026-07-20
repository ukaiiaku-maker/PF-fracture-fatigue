from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_v10_2_17_stage3_temperature_metrics.py"
SPEC = importlib.util.spec_from_file_location("v10217_plotter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def test_extracts_initial_end_and_path_length_slope(tmp_path: Path):
    case = tmp_path / "ceramic_primary" / "T700K_th45_seed2420"
    case.mkdir(parents=True)
    (case / "summary.json").write_text(json.dumps([{
        "T": 700.0,
        "Kc_first_MPa_sqrt_m": 10.0,
        "n_advances": 2,
    }]))
    (case / "v10_2_17_parameter_selection.json").write_text(json.dumps({
        "option_key": "ceramic_primary",
        "candidate_id": "ceramic_restart02_candidate00",
    }))
    (case / "stage3_case_status.json").write_text(json.dumps({
        "status": "complete_target_extension",
        "projected_extension_um": 80.0,
    }))
    (case / "v10_2_17_final_signed_stochastic_stack.json").write_text(json.dumps({
        "cleavage_hazard_seed": 2420,
    }))
    # Two 3-4-5 segments: cumulative geometric extension is 100 um, while the
    # projected x-extension is only 60 um.  This proves the slope uses path length.
    (case / "crack_path_700K.csv").write_text(
        "x_m,y_m\n"
        "0.001000,0.000000\n"
        "0.001030,0.000040\n"
        "0.001060,0.000080\n"
    )
    header = (
        "step,Uapp_m,Ftop_N,KJ_Pa_sqrtm,sigma_tip_Pa,sigma_back_Pa,"
        "lambda_c,lambda_e,B,N_em,a_tip_m,crack_extension_m,da_block_m,"
        "W_emit_J_per_m,n_fire\n"
    )
    (case / "steps_0700K.csv").write_text(
        header
        + "1,0,0,10000000,0,0,0,0,0,0,0.00103,0.00003,0.00003,0,1\n"
        + "2,0,0,12000000,0,0,0,0,0,0,0.00106,0.00006,0.00003,0,1\n"
    )

    metric, events = PLOTTER.analyze_case(case)
    assert metric.K_initial_MPa_sqrt_m == 10.0
    assert metric.K_end_MPa_sqrt_m == 12.0
    assert np.isclose(metric.path_extension_um, 100.0)
    assert metric.n_event_points == 2
    assert metric.event_mapping_mode == "n_fire_exact"
    assert len(events) == 2
    # Fit through (0,10), (50,10), (100,12).
    expected_slope = np.polyfit([0.0, 50.0, 100.0], [10.0, 10.0, 12.0], 1)[0]
    assert np.isclose(metric.Rcurve_slope_MPa_sqrt_m_per_um, expected_slope)
    assert np.isclose(metric.Rcurve_slope_MPa_sqrt_m_per_100um, 100.0 * expected_slope)
