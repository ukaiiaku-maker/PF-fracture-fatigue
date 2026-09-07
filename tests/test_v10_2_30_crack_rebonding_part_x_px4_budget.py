"""PX4: the developed-campaign trajectory budget (mission section 8 -- 30
accepted events, 150 um, or 1e12 cycles, whichever comes first).

Regression test for a real bug caught by the controller's own PX4 launch
attempt: part_x_run_one_job.py's real-budget branch always used PX3's
screen budget (12 events/60um, no cycle cap), even for developed ("D...")
protocol rows -- meaning the first PX4 launch attempt would have silently
produced PX3-screen-sized trajectories mislabeled as developed data. Fixed
by branching on job["protocol"] and adding run_trajectory's new
max_cumulative_cycles parameter (default math.inf, zero effect on every
existing caller).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)
import part_x_run_one_job as runner  # noqa: E402


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


@pytest.mark.parametrize("protocol,expected", [
    ("7.1_R_panel", False), ("7.6_cohesive_strength", False),
    ("D1", True), ("D6_conditional_persistent", True), ("D7_conditional_cohesive_strength", True),
])
def test_is_developed_protocol(protocol, expected):
    assert runner._is_developed_protocol(protocol) is expected


def test_run_trajectory_default_max_cumulative_cycles_is_infinite_and_inert():
    Engine.configure_hazard(mode="exponential", seed=1720)
    Engine.reset_audit()
    result = pilot.run_trajectory(
        name="cycles_default", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=None, R=-0.5,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
        max_accepted_events=1, max_projected_extension_m=1.0e300, hazard_rng_seed=1720,
    )
    assert not result["censored"]
    assert result["cumulative_cycles"] > 0.0


def test_run_trajectory_cycle_budget_censors_correctly():
    Engine.configure_hazard(mode="exponential", seed=1720)
    Engine.reset_audit()
    # The cap is checked at the START of each event's search (same as
    # max_accepted_events/max_projected_extension_m), so a cap tiny enough
    # to be exceeded by the first event's own accumulated cycles still
    # lets that first event complete -- it censors starting the NEXT one.
    result = pilot.run_trajectory(
        name="cycles_capped", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=None, R=-0.5,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
        max_accepted_events=1000, max_projected_extension_m=1.0e300, hazard_rng_seed=1720,
        max_cumulative_cycles=1.0e-9,
    )
    assert result["censored"] is True
    assert result["censor_reason"] == "cycle_budget_exhausted"
    assert result["n_accepted_events"] == 1
    assert result["cumulative_cycles"] >= 1.0e-9


def test_run_one_job_real_budget_selects_developed_settings_for_D_protocols(tmp_path, monkeypatch):
    """End-to-end: a job with protocol='D1' must use the developed budget
    (30/150um/1e12 cycles), not the screen budget (12/60um) -- shrunk here
    via monkeypatch so the test completes quickly, but the SELECTION
    LOGIC (which constants get used) is exercised for real."""
    import csv
    with open(REPO_ROOT / "artifacts/crack_rebonding_part_x_v1/developed_job_registry.csv") as f:
        rows = list(csv.DictReader(f))
    job = next(r for r in rows if r["protocol"] == "D1" and r["status"] == "AUTHORIZED_PX4" and r["cohesion"] == "finite")

    orig_events, orig_extension, orig_cycles = (
        runner.DEVELOPED_MAX_ACCEPTED_EVENTS, runner.DEVELOPED_MAX_PROJECTED_EXTENSION_m, runner.DEVELOPED_MAX_CUMULATIVE_CYCLES,
    )
    runner.DEVELOPED_MAX_ACCEPTED_EVENTS = 1
    runner.DEVELOPED_MAX_PROJECTED_EXTENSION_m = 1.0e300
    runner.DEVELOPED_MAX_CUMULATIVE_CYCLES = math.inf
    try:
        result_dir = tmp_path / "result"
        job_json = tmp_path / "job.json"
        import json
        job_json.write_text(json.dumps(job))
        old_argv = sys.argv
        sys.argv = ["part_x_run_one_job.py", "--job-json", str(job_json), "--result-dir", str(result_dir)]
        try:
            rc = runner.main()
        finally:
            sys.argv = old_argv
    finally:
        (runner.DEVELOPED_MAX_ACCEPTED_EVENTS, runner.DEVELOPED_MAX_PROJECTED_EXTENSION_m,
         runner.DEVELOPED_MAX_CUMULATIVE_CYCLES) = (orig_events, orig_extension, orig_cycles)

    assert rc == 0
    import json
    result = json.loads((result_dir / "result.json").read_text())
    assert result["schema"] == "v10230_part_x_developed_job_result_v1"
    assert result["trajectory"]["n_accepted_events"] == 1
