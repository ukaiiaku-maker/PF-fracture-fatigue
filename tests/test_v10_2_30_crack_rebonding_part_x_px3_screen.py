"""PX3: the real screen budget entry point (scripts/part_x_run_one_job.py)
and its two prerequisite extensions to the qualified causal-pilot-v2 event
loop (arrhenius_fracture/crack_rebonding_causal_pilot_v2_v10230.run_trajectory):

1. ``minimum_load_hold_s`` threading, so the PX3 dwell panel (mission
   section 7.3) can drive the same qualified event loop instead of a
   second reimplementation.
2. ``cumulative_cycles`` bookkeeping, needed for mission section 7.8's
   ``g = sum(accepted crack advance) / sum(physical cycles)``.

Also covers scripts/part_x_run_one_job.py's chemistry_factor /
K_rebond_max_target_Pa_sqrt_m screen-panel override resolution (sections
7.5/7.6), which must reproduce build_part_x_kinetic_regime_registry.py's
own K_target -> restored_work_of_separation_J_m2 mapping exactly.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import part_x_run_one_job as runner  # noqa: E402

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine  # noqa: E402
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform  # noqa: E402
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

SCREEN_REGISTRY_PATH = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1" / "screen_job_registry.csv"


def _authorized_jobs() -> list[dict]:
    with SCREEN_REGISTRY_PATH.open() as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r["status"] == "AUTHORIZED_PX3"]


def _make_controller(n_phase: int):
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )


def _build_engine(cfg):
    return build_a_native_engine(cfg)


def test_run_trajectory_default_hold_is_zero_and_threaded_through():
    """Default minimum_load_hold_s=0.0 must be byte-identical to the
    pre-PX3 behavior; a nonzero value must reach the actual waveform."""
    Engine.configure_hazard(mode="exponential", seed=1720)
    Engine.reset_audit()
    result_zero = pilot.run_trajectory(
        name="hold_zero", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=None, R=-0.5,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
        max_accepted_events=1, max_projected_extension_m=1.0e300,
        hazard_rng_seed=1720,
    )
    assert result_zero["minimum_load_hold_s"] == 0.0
    assert result_zero["cumulative_cycles"] > 0.0
    assert result_zero["events"][0]["cumulative_cycles"] > 0.0


def test_run_trajectory_nonzero_hold_reaches_the_waveform():
    captured: list[float] = []
    original_waveform_cls = FatigueWaveform

    def _recording_waveform_cls(**kwargs):
        captured.append(kwargs.get("minimum_load_hold_s"))
        return original_waveform_cls(**kwargs)

    Engine.configure_hazard(mode="exponential", seed=1720)
    Engine.reset_audit()
    pilot.run_trajectory(
        name="hold_nonzero", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=_recording_waveform_cls, rebonding_cfg=None, R=-0.5,
        reset_engine_registry=Engine.reset_audit,
        Kmax_Pa_sqrt_m=18.0e6, frequency_Hz=1000.0, n_phase=80, T_K_=300.0,
        max_accepted_events=1, max_projected_extension_m=1.0e300,
        hazard_rng_seed=1720, minimum_load_hold_s=0.0005,
    )
    assert captured == [0.0005]


def test_resolve_screen_rebonding_cfg_zero_cohesion_forces_zero_G():
    jobs = _authorized_jobs()
    job = next(j for j in jobs if j["protocol"] == "7.1_R_panel" and j["cohesion"] == "zero")
    row_payload = runner._load_row_payload(job)
    cfg = runner._resolve_screen_rebonding_cfg(job, row_payload, Eprime_Pa=1.0e11)
    assert cfg.restored_work_of_separation_J_m2 == 0.0
    assert cfg.chemistry_factor == 1.0


def test_resolve_screen_rebonding_cfg_reproduces_registry_K_target_mapping():
    """The G_max mapping (G = (K_target / rebond_K_geometry_factor)**2 /
    Eprime_Pa) must exactly match build_part_x_kinetic_regime_registry.py's
    own formula -- verified here by checking the two cohesive-strength
    screen jobs (K_target=0.45/1.80 MPa sqrt(m)) yield restored_work values
    in the expected ratio (0.45/1.80)**2 = 1/16, independent of Eprime_Pa."""
    jobs = _authorized_jobs()
    job_low = next(j for j in jobs if j["protocol"] == "7.6_cohesive_strength" and j["K_rebond_max_target_Pa_sqrt_m"] == "450000.0")
    job_high = next(j for j in jobs if j["protocol"] == "7.6_cohesive_strength" and j["K_rebond_max_target_Pa_sqrt_m"] == "1800000.0")
    row_payload = runner._load_row_payload(job_low)
    Eprime_Pa = 1.23e11
    cfg_low = runner._resolve_screen_rebonding_cfg(job_low, row_payload, Eprime_Pa)
    cfg_high = runner._resolve_screen_rebonding_cfg(job_high, row_payload, Eprime_Pa)
    ratio = cfg_low.restored_work_of_separation_J_m2 / cfg_high.restored_work_of_separation_J_m2
    assert ratio == pytest.approx((450000.0 / 1800000.0) ** 2, rel=1.0e-12)
    expected_low_G = (450000.0 / cfg_low.rebond_K_geometry_factor) ** 2 / Eprime_Pa
    assert cfg_low.restored_work_of_separation_J_m2 == pytest.approx(expected_low_G, rel=1.0e-12)
    K_recovered = cfg_low.rebond_K_geometry_factor * (Eprime_Pa * cfg_low.restored_work_of_separation_J_m2) ** 0.5
    assert K_recovered == pytest.approx(450000.0, rel=1.0e-9)


def test_resolve_screen_rebonding_cfg_applies_chemistry_factor_override():
    jobs = _authorized_jobs()
    job_default = next(j for j in jobs if j["protocol"] == "7.5_passivation_chemistry" and j["chemistry_factor"] == "1.0" and j["cohesion"] == "finite")
    job_low_chem = next(j for j in jobs if j["protocol"] == "7.5_passivation_chemistry" and j["chemistry_factor"] == "0.1" and j["cohesion"] == "finite")
    row_payload = runner._load_row_payload(job_default)
    cfg_default = runner._resolve_screen_rebonding_cfg(job_default, row_payload, Eprime_Pa=1.0e11)
    cfg_low_chem = runner._resolve_screen_rebonding_cfg(job_low_chem, row_payload, Eprime_Pa=1.0e11)
    assert cfg_default.chemistry_factor == 1.0
    assert cfg_low_chem.chemistry_factor == 0.1
    assert cfg_default.config_hash() != cfg_low_chem.config_hash()


def test_run_one_job_real_budget_end_to_end(tmp_path):
    """Full end-to-end smoke test of part_x_run_one_job.py's real (non-
    preflight) branch against an actual AUTHORIZED_PX3 screen row, with the
    screen budget shrunk to 1 accepted event so the test completes quickly.
    Restores the module's screen-budget constants afterward so no other
    test observes the shrunk budget."""
    jobs = _authorized_jobs()
    job = next(j for j in jobs if j["protocol"] == "7.1_R_panel" and j["cohesion"] == "finite")

    orig_events, orig_extension = runner.SCREEN_MAX_ACCEPTED_EVENTS, runner.SCREEN_MAX_PROJECTED_EXTENSION_m
    runner.SCREEN_MAX_ACCEPTED_EVENTS = 1
    runner.SCREEN_MAX_PROJECTED_EXTENSION_m = 1.0e300
    try:
        result_dir = tmp_path / "result"
        job_json = tmp_path / "job.json"
        job_json.write_text(json.dumps(job))
        old_argv = sys.argv
        sys.argv = ["part_x_run_one_job.py", "--job-json", str(job_json), "--result-dir", str(result_dir)]
        try:
            rc = runner.main()
        finally:
            sys.argv = old_argv
    finally:
        runner.SCREEN_MAX_ACCEPTED_EVENTS = orig_events
        runner.SCREEN_MAX_PROJECTED_EXTENSION_m = orig_extension

    assert rc == 0
    status = json.loads((result_dir / "job_status.json").read_text())
    assert status["status"] == "COMPLETE"
    result = json.loads((result_dir / "result.json").read_text())
    assert result["preflight"] is False
    assert result["schema"] == "v10230_part_x_screen_job_result_v1"
    traj = result["trajectory"]
    assert traj["n_accepted_events"] == 1
    assert not traj["censored"]
    assert traj["cumulative_extension_m"] > 0.0
    assert traj["cumulative_cycles"] > 0.0
    assert traj["minimum_load_hold_s"] == float(job["minimum_load_hold_s"])
